export interface Chunk {
  text: string;
  arxiv_id: string;
  section: string;
  chunk_index: number;
  score: number;
  title: string | null;
  authors: string | null;
  year: string | number | null;
}

export interface ConfigInfo {
  name: string;
  description: string;
}

export interface ConfigMetrics {
  faithfulness: number;
  context_precision: number;
  context_recall: number;
  answer_relevancy: number;
}

export interface EvalResults {
  timestamp: string | null;
  dataset_size: number | null;
  configs: Record<string, ConfigMetrics>;
}

export type QueryStatus = 'idle' | 'streaming' | 'done' | 'error';
