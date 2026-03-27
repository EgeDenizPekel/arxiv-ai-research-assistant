import { AlertCircle } from 'lucide-react';
import type { QueryStatus } from '../types';

interface AnswerPanelProps {
  answer: string;
  status: QueryStatus;
  error: string | null;
}

export function AnswerPanel({ answer, status, error }: AnswerPanelProps) {
  if (status === 'idle') return null;

  return (
    <div className="bg-[#1a1d2e] border border-[#2a2d3e] rounded-xl p-5">
      <h2 className="text-sm font-semibold text-[#818cf8] uppercase tracking-widest mb-4">
        Answer
      </h2>

      {error ? (
        <div className="flex items-start gap-3 text-red-400 text-sm">
          <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      ) : (
        <div
          className={`text-sm text-[#e2e8f0] leading-relaxed whitespace-pre-wrap min-h-[60px] ${
            status === 'streaming' ? 'streaming-cursor' : ''
          }`}
        >
          {answer || (
            <span className="text-[#4a5166] italic">Retrieving context...</span>
          )}
        </div>
      )}
    </div>
  );
}
