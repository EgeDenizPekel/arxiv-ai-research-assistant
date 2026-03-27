import os
from typing import AsyncGenerator

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from src.retrieval import get_retriever

load_dotenv()

MODEL = "gpt-4o-mini"
DEFAULT_CONFIG = "reranked"
DEFAULT_TOP_K = 5

SYSTEM_PROMPT = """\
You are a research assistant specializing in machine learning and AI.
Answer the user's question using ONLY the research paper excerpts provided below.

Rules:
- If the answer is present in the excerpts, answer directly and cite the source by its ID (e.g. "According to [hf_abc123]...").
- If the excerpts do not contain enough information to answer, say so clearly. Do not speculate or make up information.
- Keep your answer concise and grounded in the provided text.

Research paper excerpts:
{context}"""

HUMAN_PROMPT = "{question}"


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    if not chunks:
        return "No relevant excerpts found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("arxiv_id", "unknown")
        section = chunk.get("section", "")
        text = chunk.get("text", "").strip()
        header = f"[{i}] ID: {source}" + (f" | Section: {section}" if section else "")
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


def _build_chain(streaming: bool = True) -> object:
    """Build the LangChain chain."""
    llm = ChatOpenAI(
        model=MODEL,
        streaming=streaming,
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    return prompt | llm | StrOutputParser()


async def stream_rag_response(
    query: str,
    config: str = DEFAULT_CONFIG,
    top_k: int = DEFAULT_TOP_K,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that yields typed SSE events:

        {"type": "sources", "chunks": [...]}  - immediately after retrieval
        {"type": "token",   "content": "..."}  - one per streamed token
        {"type": "done"}                        - signals stream end

    Args:
        query:  The user's question.
        config: Retriever config name ('naive', 'hybrid', 'reranked', 'hyde').
        top_k:  Number of chunks to retrieve.
    """
    # --- Retrieval ---
    retriever = get_retriever(config)
    logger.info(f"[{config}] Retrieving for: {query[:80]}")
    chunks = retriever.retrieve(query, top_k=top_k)
    logger.info(f"[{config}] Retrieved {len(chunks)} chunks")

    yield {"type": "sources", "chunks": chunks}

    if not chunks:
        yield {"type": "token", "content": "I couldn't find relevant information in the corpus to answer this question."}
        yield {"type": "done"}
        return

    # --- Generation ---
    context = _format_context(chunks)
    chain = _build_chain(streaming=True)

    logger.info(f"[{config}] Streaming answer from {MODEL}")
    async for token in chain.astream({"context": context, "question": query}):
        yield {"type": "token", "content": token}

    yield {"type": "done"}


def get_rag_response(
    query: str,
    config: str = DEFAULT_CONFIG,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Synchronous (non-streaming) RAG response. Used by the evaluation runner.

    Returns:
        {
            "answer":  str,
            "chunks":  list[dict],
            "config":  str,
        }
    """
    retriever = get_retriever(config)
    chunks = retriever.retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the corpus to answer this question.",
            "chunks": [],
            "config": config,
        }

    context = _format_context(chunks)
    chain = _build_chain(streaming=False)
    answer = chain.invoke({"context": context, "question": query})

    return {"answer": answer, "chunks": chunks, "config": config}
