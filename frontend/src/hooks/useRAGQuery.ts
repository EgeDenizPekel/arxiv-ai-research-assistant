import { useState, useCallback, useRef } from 'react';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import type { Chunk, QueryStatus } from '../types';

interface UseRAGQueryReturn {
  answer: string;
  chunks: Chunk[];
  status: QueryStatus;
  error: string | null;
  submit: (query: string, config: string, topK: number) => void;
  reset: () => void;
}

export function useRAGQuery(): UseRAGQueryReturn {
  const [answer, setAnswer] = useState('');
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [status, setStatus] = useState<QueryStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setAnswer('');
    setChunks([]);
    setStatus('idle');
    setError(null);
  }, []);

  const submit = useCallback((query: string, config: string, topK: number) => {
    // Abort any in-flight request
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setAnswer('');
    setChunks([]);
    setError(null);
    setStatus('streaming');

    fetchEventSource('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, config, top_k: topK }),
      signal: controller.signal,

      onmessage(ev) {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === 'sources') {
            setChunks(data.chunks);
          } else if (data.type === 'token') {
            setAnswer(prev => prev + data.content);
          } else if (data.type === 'done') {
            setStatus('done');
          } else if (data.type === 'error') {
            setError(data.message);
            setStatus('error');
          }
        } catch {
          // ignore parse errors on keep-alive newlines
        }
      },

      onerror(err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setError('Connection error. Is the API running?');
        setStatus('error');
        throw err; // stops retries
      },

      onclose() {
        if (status !== 'error') {
          setStatus('done');
        }
      },
    });
  }, [status]);

  return { answer, chunks, status, error, submit, reset };
}
