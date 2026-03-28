import { useState, type KeyboardEvent } from 'react';
import { Search, Loader2 } from 'lucide-react';
import type { ConfigInfo, QueryStatus } from '../types';

interface QueryPanelProps {
  configs: ConfigInfo[];
  status: QueryStatus;
  onSubmit: (query: string, config: string, topK: number) => void;
  onReset: () => void;
}

const EXAMPLE_QUESTIONS = [
  'How does self-attention work?',
  'What is masked language modeling?',
  'Explain variational autoencoders',
  'How do graph neural networks work?',
  'What is contrastive learning?',
  'How does RLHF fine-tune LLMs?',
];

const TOP_K_PRESETS = [3, 5, 10];

export function QueryPanel({ configs, status, onSubmit, onReset }: QueryPanelProps) {
  const [query, setQuery] = useState('');
  const [config, setConfig] = useState('reranked');
  const [topK, setTopK] = useState(5);

  const isLoading = status === 'streaming';
  const selectedConfig = configs.find(c => c.name === config);

  const handleSubmit = () => {
    if (!query.trim() || isLoading) return;
    onSubmit(query.trim(), config, topK);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit();
  };

  const handleReset = () => {
    setQuery('');
    onReset();
  };

  return (
    <div className="bg-[#1a1d2e] border border-[#2a2d3e] rounded-xl p-5">
      <h2 className="text-sm font-semibold text-[#818cf8] uppercase tracking-widest mb-1">
        Ask a Question
      </h2>
      <p className="text-xs text-[#6b7280] mb-4">
        Ask anything about ML/AI research. The system retrieves relevant paper chunks and generates a grounded answer.
      </p>

      {/* Example questions */}
      <div className="mb-3">
        <p className="text-xs text-[#4a5166] mb-2">Try an example:</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLE_QUESTIONS.map(q => (
            <button
              key={q}
              onClick={() => setQuery(q)}
              className="text-xs bg-[#0f1117] border border-[#2a2d3e] hover:border-[#818cf8] text-[#94a3b8] hover:text-[#c4b5fd] px-2.5 py-1 rounded-full transition-colors whitespace-nowrap"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Query textarea */}
      <textarea
        value={query}
        onChange={e => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type your question here... (Cmd+Enter to submit)"
        rows={4}
        className="w-full bg-[#0f1117] border border-[#2a2d3e] rounded-lg px-4 py-3 text-sm text-[#e2e8f0] placeholder-[#4a5166] resize-none focus:outline-none focus:border-[#818cf8] transition-colors"
      />

      {/* Retrieval strategy */}
      <div className="mt-3">
        <label className="block text-xs text-[#6b7280] mb-1">Retrieval strategy</label>
        <select
          value={config}
          onChange={e => setConfig(e.target.value)}
          className="w-full bg-[#0f1117] border border-[#2a2d3e] rounded-lg px-3 py-2 text-sm text-[#e2e8f0] focus:outline-none focus:border-[#818cf8] transition-colors"
        >
          {(configs.length > 0 ? configs : [
            { name: 'naive',    description: '' },
            { name: 'hybrid',   description: '' },
            { name: 'reranked', description: '' },
            { name: 'hyde',     description: '' },
          ]).map(c => (
            <option key={c.name} value={c.name}>
              {c.name === 'naive'    ? 'Naive - dense only'               :
               c.name === 'hybrid'   ? 'Hybrid - dense + BM25'            :
               c.name === 'reranked' ? 'Reranked - hybrid + cross-encoder' :
               c.name === 'hyde'     ? 'HyDE - hypothetical doc'          : c.name}
            </option>
          ))}
        </select>
        {selectedConfig?.description && (
          <p className="text-xs text-[#4a5166] mt-1.5 leading-relaxed">
            {selectedConfig.description}
          </p>
        )}
      </div>

      {/* Top-K slider */}
      <div className="mt-4">
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs text-[#6b7280]">
            Top-K chunks
            <span className="text-[#4a5166] ml-1">- how many paper chunks the LLM receives as context</span>
          </label>
          <span className="text-xs font-mono font-semibold text-[#c4b5fd] bg-[#818cf8]/10 border border-[#818cf8]/20 px-2 py-0.5 rounded">
            {topK}
          </span>
        </div>

        {/* Preset chips */}
        <div className="flex items-center gap-2 mb-2">
          {TOP_K_PRESETS.map(v => (
            <button
              key={v}
              onClick={() => setTopK(v)}
              className={`text-xs px-2.5 py-0.5 rounded-full border transition-colors ${
                topK === v
                  ? 'bg-[#818cf8]/20 border-[#818cf8]/50 text-[#c4b5fd]'
                  : 'bg-[#0f1117] border-[#2a2d3e] text-[#4a5166] hover:border-[#818cf8]/40 hover:text-[#94a3b8]'
              }`}
            >
              {v}
            </button>
          ))}
        </div>

        <input
          type="range"
          min={1}
          max={20}
          value={topK}
          onChange={e => setTopK(Number(e.target.value))}
          className="w-full accent-[#818cf8] h-1.5 rounded-full cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-[#3a3f55] mt-1">
          <span>1 - faster</span>
          <span>20 - more context</span>
        </div>
      </div>

      {/* Buttons */}
      <div className="flex gap-2 mt-5">
        <button
          onClick={handleSubmit}
          disabled={!query.trim() || isLoading}
          className="flex items-center gap-2 bg-[#818cf8] hover:bg-[#6366f1] disabled:bg-[#2a2d3e] disabled:text-[#4a5166] disabled:cursor-not-allowed text-white text-sm font-medium px-5 py-2.5 rounded-lg transition-colors"
        >
          {isLoading ? (
            <Loader2 size={15} className="animate-spin" />
          ) : (
            <Search size={15} />
          )}
          {isLoading ? 'Retrieving & generating...' : 'Ask'}
        </button>

        {status !== 'idle' && (
          <button
            onClick={handleReset}
            className="text-sm text-[#6b7280] hover:text-[#e2e8f0] px-4 py-2.5 rounded-lg border border-[#2a2d3e] hover:border-[#4a5166] transition-colors"
          >
            Clear
          </button>
        )}
      </div>
    </div>
  );
}
