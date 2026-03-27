import { useState, type KeyboardEvent } from 'react';
import { Search, Loader2, Info } from 'lucide-react';
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

const TOP_K_DESCRIPTION =
  'Number of paper chunks retrieved and passed to the LLM as context. Higher = more context, but slower and more expensive.';

export function QueryPanel({ configs, status, onSubmit, onReset }: QueryPanelProps) {
  const [query, setQuery] = useState('');
  const [config, setConfig] = useState('reranked');
  const [topK, setTopK] = useState(5);
  const [showTopKTip, setShowTopKTip] = useState(false);

  const isLoading = status === 'streaming';

  const selectedConfig = configs.find(c => c.name === config);

  const handleSubmit = () => {
    if (!query.trim() || isLoading) return;
    onSubmit(query.trim(), config, topK);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
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

      {/* Config + top-k row */}
      <div className="flex gap-3 mt-3">
        <div className="flex-1">
          <label className="block text-xs text-[#6b7280] mb-1">
            Retrieval strategy
          </label>
          <select
            value={config}
            onChange={e => setConfig(e.target.value)}
            className="w-full bg-[#0f1117] border border-[#2a2d3e] rounded-lg px-3 py-2 text-sm text-[#e2e8f0] focus:outline-none focus:border-[#818cf8] transition-colors"
          >
            {(configs.length > 0 ? configs : [
              { name: 'naive', description: '' },
              { name: 'hybrid', description: '' },
              { name: 'reranked', description: '' },
              { name: 'hyde', description: '' },
            ]).map(c => (
              <option key={c.name} value={c.name}>
                {c.name === 'naive'    ? 'Naive - dense only'             :
                 c.name === 'hybrid'   ? 'Hybrid - dense + BM25'          :
                 c.name === 'reranked' ? 'Reranked - hybrid + cross-encoder' :
                 c.name === 'hyde'     ? 'HyDE - hypothetical doc'        : c.name}
              </option>
            ))}
          </select>
          {/* Config description */}
          {selectedConfig?.description && (
            <p className="text-xs text-[#4a5166] mt-1.5 leading-relaxed">
              {selectedConfig.description}
            </p>
          )}
        </div>

        <div className="w-24">
          <label className="flex items-center gap-1 text-xs text-[#6b7280] mb-1">
            Top-K chunks
            <button
              onMouseEnter={() => setShowTopKTip(true)}
              onMouseLeave={() => setShowTopKTip(false)}
              className="text-[#4a5166] hover:text-[#818cf8] transition-colors"
            >
              <Info size={11} />
            </button>
          </label>
          <input
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={e => setTopK(Number(e.target.value))}
            className="w-full bg-[#0f1117] border border-[#2a2d3e] rounded-lg px-3 py-2 text-sm text-[#e2e8f0] focus:outline-none focus:border-[#818cf8] transition-colors"
          />
          {showTopKTip && (
            <div className="absolute z-20 mt-1 w-56 bg-[#1e2130] border border-[#2a2d3e] rounded-lg p-2.5 text-xs text-[#94a3b8] shadow-xl">
              {TOP_K_DESCRIPTION}
            </div>
          )}
        </div>
      </div>

      {/* Buttons */}
      <div className="flex gap-2 mt-4">
        <button
          onClick={handleSubmit}
          disabled={!query.trim() || isLoading}
          className="flex items-center gap-2 bg-[#818cf8] hover:bg-[#6366f1] disabled:bg-[#3a3f55] disabled:cursor-not-allowed text-white text-sm font-medium px-5 py-2.5 rounded-lg transition-colors"
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
