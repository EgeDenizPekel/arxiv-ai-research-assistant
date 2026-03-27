import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { EvalResults } from '../types';

interface MetricsChartProps {
  results: EvalResults;
}

const METRIC_COLORS: Record<string, string> = {
  faithfulness: '#818cf8',
  context_precision: '#34d399',
  context_recall: '#f59e0b',
  answer_relevancy: '#f472b6',
};

const METRIC_LABELS: Record<string, string> = {
  faithfulness: 'Faithfulness',
  context_precision: 'Ctx Precision',
  context_recall: 'Ctx Recall',
  answer_relevancy: 'Ans Relevancy',
};

const METRIC_DESCRIPTIONS: Record<string, string> = {
  faithfulness:
    'Are claims in the answer actually supported by the retrieved context? Catches hallucination.',
  context_precision:
    'Are the top-ranked retrieved chunks actually relevant to the question? Measures retrieval quality.',
  context_recall:
    'Did retrieval find all the chunks needed to fully answer the question? Measures coverage.',
  answer_relevancy:
    'Does the answer directly address what was asked? Penalises vague or off-topic responses.',
};

const CONFIG_DESCRIPTIONS: Record<string, string> = {
  naive: 'Dense-only vector search. Fast baseline with no reranking.',
  hybrid: 'Combines dense embeddings with BM25 keyword search via Reciprocal Rank Fusion.',
  reranked: 'Hybrid retrieval re-scored by a cross-encoder for higher precision.',
  hyde: 'GPT-4o-mini generates a hypothetical answer first, then searches using its embedding.',
};

const CONFIG_ORDER = ['naive', 'hybrid', 'reranked', 'hyde'];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1e2130] border border-[#2a2d3e] rounded-lg p-3 shadow-xl">
      <p className="text-xs font-semibold text-[#e2e8f0] mb-1 capitalize">{label}</p>
      {CONFIG_DESCRIPTIONS[label] && (
        <p className="text-xs text-[#4a5166] mb-2 max-w-[200px]">{CONFIG_DESCRIPTIONS[label]}</p>
      )}
      {payload.map((entry: any) => (
        <div key={entry.name} className="flex items-center gap-2 text-xs">
          <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: entry.fill }} />
          <span className="text-[#94a3b8]">{METRIC_LABELS[entry.name] ?? entry.name}:</span>
          <span className="text-[#e2e8f0] font-mono">{(entry.value * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
};

export function MetricsChart({ results }: MetricsChartProps) {
  const data = CONFIG_ORDER
    .filter(cfg => cfg in results.configs)
    .map(cfg => ({
      config: cfg,
      ...results.configs[cfg],
    }));

  const metrics = ['faithfulness', 'context_precision', 'context_recall', 'answer_relevancy'];

  return (
    <div className="bg-[#1a1d2e] border border-[#2a2d3e] rounded-xl p-5">
      {/* Header */}
      <div className="flex items-start justify-between mb-1">
        <h2 className="text-sm font-semibold text-[#818cf8] uppercase tracking-widest">
          RAGAS Evaluation
        </h2>
        {results.timestamp && (
          <span className="text-xs text-[#4a5166]">{results.timestamp}</span>
        )}
      </div>
      <p className="text-xs text-[#6b7280] mb-1">
        Each retrieval strategy was evaluated on {results.dataset_size ?? '?'} questions using RAGAS - an LLM-as-judge framework that scores both retrieval quality and answer quality.
      </p>
      <p className="text-xs text-[#4a5166] mb-5">
        Scores closer to 100% are better. <span className="text-emerald-400">Green</span> cells are the best in each column.
      </p>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: 4, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2130" />
          <XAxis
            dataKey="config"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={{ stroke: '#2a2d3e' }}
            tickLine={false}
          />
          <YAxis
            domain={[0.75, 1.0]}
            tickFormatter={v => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: '#94a3b8', fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
          <Legend
            formatter={(value) => (
              <span style={{ color: '#94a3b8', fontSize: 11 }}>
                {METRIC_LABELS[value] ?? value}
              </span>
            )}
          />
          {metrics.map(metric => (
            <Bar
              key={metric}
              dataKey={metric}
              fill={METRIC_COLORS[metric]}
              radius={[3, 3, 0, 0]}
              maxBarSize={28}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>

      {/* Score table */}
      <div className="mt-5 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-[#4a5166] border-b border-[#2a2d3e]">
              <th className="text-left py-2 pr-4 font-medium">Strategy</th>
              {metrics.map(m => (
                <th key={m} className="text-right py-2 px-2 font-medium">
                  {METRIC_LABELS[m]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map(row => (
              <tr key={row.config} className="border-b border-[#1e2130] hover:bg-[#0f1117] transition-colors">
                <td className="py-3 pr-4">
                  <span className="text-[#c4b5fd] font-medium capitalize">{row.config}</span>
                </td>
                {metrics.map(m => {
                  const val = (row as any)[m] as number;
                  const best = Math.max(...data.map(r => (r as any)[m]));
                  return (
                    <td
                      key={m}
                      className={`text-right py-2 px-2 font-mono align-top ${
                        val === best ? 'text-emerald-400 font-semibold' : 'text-[#94a3b8]'
                      }`}
                    >
                      {(val * 100).toFixed(1)}%
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Metric definitions */}
      <div className="mt-6 border-t border-[#1e2130] pt-5">
        <p className="text-xs font-semibold text-[#4a5166] uppercase tracking-wider mb-3">
          What do these metrics mean?
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {metrics.map(m => (
            <div key={m} className="flex gap-2">
              <span
                className="w-2 h-2 rounded-full flex-shrink-0 mt-1"
                style={{ background: METRIC_COLORS[m] }}
              />
              <div>
                <span className="text-xs font-medium text-[#94a3b8]">{METRIC_LABELS[m]}: </span>
                <span className="text-xs text-[#4a5166]">{METRIC_DESCRIPTIONS[m]}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
