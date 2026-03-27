import { useState } from 'react';
import { ChevronDown, ChevronUp, FileText } from 'lucide-react';
import type { Chunk } from '../types';

interface SourcesPanelProps {
  chunks: Chunk[];
}

function ScoreBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color =
    pct >= 80 ? 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30' :
    pct >= 60 ? 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30' :
                'text-slate-400 bg-slate-400/10 border-slate-400/30';
  return (
    <span className={`text-xs font-mono px-2 py-0.5 rounded border ${color}`}>
      {pct}%
    </span>
  );
}

function ChunkCard({ chunk, index }: { chunk: Chunk; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const preview = chunk.text.slice(0, 220);
  const hasMore = chunk.text.length > 220;

  return (
    <div className="bg-[#0f1117] border border-[#2a2d3e] rounded-lg p-4">
      {/* Header row */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs text-[#4a5166] font-mono flex-shrink-0">#{index + 1}</span>
          <FileText size={13} className="text-[#818cf8] flex-shrink-0" />
          <span className="text-xs text-[#c4b5fd] font-medium truncate">
            {chunk.title ?? chunk.arxiv_id}
          </span>
        </div>
        <ScoreBadge score={chunk.score} />
      </div>

      {/* Section tag */}
      {chunk.section && (
        <div className="mb-2">
          <span className="text-xs bg-[#1a1d2e] border border-[#2a2d3e] text-[#6b7280] px-2 py-0.5 rounded">
            {chunk.section}
          </span>
        </div>
      )}

      {/* Text preview */}
      <p className="text-xs text-[#94a3b8] leading-relaxed">
        {expanded ? chunk.text : preview}
        {!expanded && hasMore && '...'}
      </p>

      {hasMore && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 mt-2 text-xs text-[#818cf8] hover:text-[#a5b4fc] transition-colors"
        >
          {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  );
}

export function SourcesPanel({ chunks }: SourcesPanelProps) {
  if (chunks.length === 0) return null;

  return (
    <div className="bg-[#1a1d2e] border border-[#2a2d3e] rounded-xl p-5">
      <h2 className="text-sm font-semibold text-[#818cf8] uppercase tracking-widest mb-4">
        Retrieved Sources
        <span className="ml-2 text-xs font-normal text-[#4a5166] normal-case">
          ({chunks.length} chunks)
        </span>
      </h2>

      <div className="flex flex-col gap-3">
        {chunks.map((chunk, i) => (
          <ChunkCard key={`${chunk.arxiv_id}-${chunk.chunk_index}`} chunk={chunk} index={i} />
        ))}
      </div>
    </div>
  );
}
