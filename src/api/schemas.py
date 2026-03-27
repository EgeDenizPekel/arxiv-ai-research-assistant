from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    config: str = Field("reranked", description="Retriever config: naive|hybrid|reranked|hyde")
    top_k: int = Field(5, ge=1, le=20)


class ConfigInfo(BaseModel):
    name: str
    description: str


class ChunkItem(BaseModel):
    text: str
    arxiv_id: str
    section: str
    chunk_index: int
    score: float
    title: str | None = None
    authors: str | None = None
    year: str | None = None


class PaperItem(BaseModel):
    arxiv_id: str
    title: str
    authors: str
    year: str | int


class PapersResponse(BaseModel):
    papers: list[PaperItem]
    total: int
    page: int
    page_size: int


class ConfigMetrics(BaseModel):
    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevancy: float


class EvalResultsResponse(BaseModel):
    timestamp: str | None
    dataset_size: int | None
    configs: dict[str, ConfigMetrics]
