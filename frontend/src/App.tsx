import { useEffect, useState } from 'react';
import { BookOpen, Zap, Database, GitBranch, FlaskConical, Cpu } from 'lucide-react';
import { QueryPanel } from './components/QueryPanel';
import { AnswerPanel } from './components/AnswerPanel';
import { SourcesPanel } from './components/SourcesPanel';
import { MetricsChart } from './components/MetricsChart';
import { useRAGQuery } from './hooks/useRAGQuery';
import type { ConfigInfo, EvalResults } from './types';
import './index.css';

const STAT_PILLS = [
  { icon: Database,    label: '750 ML/AI Papers' },
  { icon: GitBranch,   label: '4 Retrieval Strategies' },
  { icon: FlaskConical,label: 'RAGAS Evaluated' },
  { icon: Cpu,         label: 'GPT-4o-mini' },
];

export default function App() {
  const [configs, setConfigs] = useState<ConfigInfo[]>([]);
  const [evalResults, setEvalResults] = useState<EvalResults | null>(null);
  const [activeTab, setActiveTab] = useState<'query' | 'eval'>('query');

  const { answer, chunks, status, error, submit, reset } = useRAGQuery();

  useEffect(() => {
    fetch('/configs')
      .then(r => r.json())
      .then(setConfigs)
      .catch(() => {});

    fetch('/eval-results')
      .then(r => r.json())
      .then(setEvalResults)
      .catch(() => {});
  }, []);

  return (
    <div className="min-h-screen bg-[#0f1117] flex flex-col">

      {/* ── Header ── */}
      <header className="border-b border-[#1e2130] bg-[#0f1117]/90 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-[#818cf8]/20 border border-[#818cf8]/40 flex items-center justify-center flex-shrink-0">
              <BookOpen size={14} className="text-[#818cf8]" />
            </div>
            <span className="text-sm font-semibold text-[#e2e8f0]">ArXiv RAG Assistant</span>
          </div>

          <div className="flex items-center gap-1 bg-[#1a1d2e] border border-[#2a2d3e] rounded-lg p-1">
            <button
              onClick={() => setActiveTab('query')}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors ${
                activeTab === 'query' ? 'bg-[#818cf8] text-white' : 'text-[#6b7280] hover:text-[#e2e8f0]'
              }`}
            >
              <BookOpen size={12} /> Query
            </button>
            <button
              onClick={() => setActiveTab('eval')}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors ${
                activeTab === 'eval' ? 'bg-[#818cf8] text-white' : 'text-[#6b7280] hover:text-[#e2e8f0]'
              }`}
            >
              <Zap size={12} /> Evaluation
            </button>
          </div>
        </div>
      </header>

      {/* ── Hero ── */}
      <div className="bg-gradient-to-b from-[#161929] to-[#0f1117] border-b border-[#1e2130]">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <h2 className="text-xl font-bold text-[#e2e8f0] mb-1 tracking-tight">
            Ask questions across ML/AI research papers
          </h2>
          <p className="text-sm text-[#6b7280] mb-5">
            Retrieves the most relevant paper chunks, then generates a grounded answer using only
            what was found - no hallucination from training data.
          </p>
          <div className="flex flex-wrap gap-2">
            {STAT_PILLS.map(({ icon: Icon, label }) => (
              <span
                key={label}
                className="flex items-center gap-1.5 bg-[#1a1d2e] border border-[#2a2d3e] text-xs text-[#94a3b8] px-3 py-1.5 rounded-full"
              >
                <Icon size={11} className="text-[#818cf8]" />
                {label}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── Main ── */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8">
        {activeTab === 'query' ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">

            {/* Left: query + answer */}
            <div className="flex flex-col gap-5">
              <QueryPanel configs={configs} status={status} onSubmit={submit} onReset={reset} />
              <AnswerPanel answer={answer} status={status} error={error} />
            </div>

            {/* Right: sources or explainer */}
            <div className="lg:sticky lg:top-[68px]">
              {chunks.length > 0 ? (
                <SourcesPanel chunks={chunks} />
              ) : (
                <HowItWorks onSwitchToEval={() => setActiveTab('eval')} />
              )}
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto">
            {evalResults ? (
              <MetricsChart results={evalResults} />
            ) : (
              <div className="bg-[#1a1d2e] border border-[#2a2d3e] border-dashed rounded-xl p-12 text-center">
                <Zap size={32} className="text-[#2a2d3e] mx-auto mb-3" />
                <p className="text-sm text-[#4a5166]">No evaluation results found.</p>
                <p className="text-xs text-[#3a3f55] mt-1">
                  Run{' '}
                  <code className="bg-[#0f1117] px-1 py-0.5 rounded text-[#818cf8]">
                    poetry run python scripts/run_eval.py
                  </code>{' '}
                  to generate metrics.
                </p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-[#1e2130] mt-auto">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <span className="text-xs text-[#3a3f55]">
            ArXiv RAG Assistant - production RAG system with hybrid retrieval + RAGAS evaluation
          </span>
          <span className="text-xs text-[#3a3f55]">
            Embeddings: BAAI/bge-large-en-v1.5 · Reranker: ms-marco-MiniLM-L-6-v2
          </span>
        </div>
      </footer>
    </div>
  );
}

/* ── How It Works sidebar ── */
function HowItWorks({ onSwitchToEval }: { onSwitchToEval: () => void }) {
  const steps = [
    {
      n: '1',
      title: 'Retrieve',
      detail:
        'Your question is embedded and matched against paper chunks using the selected strategy - dense vectors, BM25 keyword search, or a fusion of both.',
    },
    {
      n: '2',
      title: 'Rerank (optional)',
      detail:
        'With Reranked or HyDE strategies, a cross-encoder model re-scores all candidates for higher precision before passing them to the LLM.',
    },
    {
      n: '3',
      title: 'Generate',
      detail:
        'GPT-4o-mini receives the top-K chunks as context and streams a grounded answer. It only uses what was retrieved - no hallucination from training data.',
    },
    {
      n: '4',
      title: 'Inspect',
      detail:
        'Source chunks appear here with relevance scores so you can trace every claim back to the original paper.',
    },
  ];

  return (
    <div className="bg-[#1a1d2e] border border-[#2a2d3e] rounded-xl overflow-hidden">
      <div className="px-5 pt-5 pb-4 border-b border-[#1e2130]">
        <h3 className="text-sm font-semibold text-[#818cf8] uppercase tracking-widest">
          How It Works
        </h3>
        <p className="text-xs text-[#6b7280] mt-1">
          Submit a question on the left to see retrieved sources here.
        </p>
      </div>

      <div className="p-5 flex flex-col gap-5">
        {steps.map(({ n, title, detail }) => (
          <div key={n} className="flex gap-3">
            <div className="w-6 h-6 rounded-full bg-[#818cf8]/15 border border-[#818cf8]/30 flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-[10px] font-bold text-[#818cf8]">{n}</span>
            </div>
            <div>
              <p className="text-xs font-semibold text-[#c4b5fd] mb-0.5">{title}</p>
              <p className="text-xs text-[#6b7280] leading-relaxed">{detail}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="px-5 pb-5 pt-1">
        <button
          onClick={onSwitchToEval}
          className="w-full text-xs text-center text-[#818cf8] hover:text-[#a5b4fc] border border-[#2a2d3e] hover:border-[#818cf8]/40 rounded-lg py-2.5 transition-colors"
        >
          Compare all 4 strategies with RAGAS metrics →
        </button>
      </div>
    </div>
  );
}
