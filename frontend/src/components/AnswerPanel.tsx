import { AlertCircle, MessageSquare } from 'lucide-react';
import type { QueryStatus } from '../types';

const CONFIG_DISPLAY: Record<string, string> = {
  naive:    'Naive',
  hybrid:   'Hybrid',
  reranked: 'Reranked',
  hyde:     'HyDE',
};

interface AnswerPanelProps {
  answer: string;
  status: QueryStatus;
  error: string | null;
  usedConfig: string | null;
}

export function AnswerPanel({ answer, status, error, usedConfig }: AnswerPanelProps) {
  // Idle: show a stabilising placeholder so the layout doesn't collapse
  if (status === 'idle') {
    return (
      <div className="bg-[#1a1d2e]/40 border border-dashed border-[#2a2d3e] rounded-xl p-5 flex items-center gap-3 min-h-[80px]">
        <MessageSquare size={16} className="text-[#2a2d3e] flex-shrink-0" />
        <p className="text-xs text-[#3a3f55]">
          Your answer will stream here after you submit a question.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-[#1a1d2e] border border-[#2a2d3e] rounded-xl p-5">
      {/* Header with strategy badge */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-[#818cf8] uppercase tracking-widest">
          Answer
        </h2>
        {usedConfig && (
          <span className="flex items-center gap-1.5 text-xs bg-[#818cf8]/10 border border-[#818cf8]/25 text-[#a5b4fc] px-2.5 py-1 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-[#818cf8]" />
            {CONFIG_DISPLAY[usedConfig] ?? usedConfig}
          </span>
        )}
      </div>

      {error ? (
        <div className="flex items-start gap-3 text-red-400 text-sm">
          <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      ) : (
        <div
          className={`text-base text-[#f1f5f9] leading-relaxed whitespace-pre-wrap min-h-[60px] ${
            status === 'streaming' ? 'streaming-cursor' : ''
          }`}
        >
          {answer || (
            <span className="text-[#4a5166] italic text-sm">
              {status === 'streaming' ? 'Retrieving context...' : ''}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
