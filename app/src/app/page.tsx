'use client';

import { useState } from 'react';
import { CheckerForm } from '../components/CheckerForm.js';
import { VerdictCard } from '../components/VerdictCard.js';
import { AgentTrace } from '../components/AgentTrace.js';
import { AuditLog } from '../components/AuditLog.js';
import {
  VerdictSchema,
  type CheckRequest,
  type ProgressEvent,
  type Verdict,
} from '../schemas/index.js';

type StreamMessage =
  | { type: 'progress'; event: ProgressEvent }
  | { type: 'verdict'; verdict: unknown }
  | { type: 'error'; status: number; error: { message: string } };

export default function Page() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [current, setCurrent] = useState<Verdict | null>(null);
  const [progress, setProgress] = useState<ProgressEvent[]>([]);
  const [history, setHistory] = useState<Verdict[]>([]);

  const handleSubmit = async (req: CheckRequest) => {
    setLoading(true);
    setError(null);
    setCurrent(null);
    setProgress([]);
    try {
      const res = await fetch('/api/check', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          accept: 'application/x-ndjson',
        },
        body: JSON.stringify(req),
      });
      if (!res.ok || !res.body) {
        const text = await res.text();
        let msg = `request failed (${res.status})`;
        try {
          const parsed = JSON.parse(text) as { error?: { message?: string } };
          if (parsed?.error?.message) msg = parsed.error.message;
        } catch {
          /* ignore */
        }
        throw new Error(msg);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let verdict: Verdict | null = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let newlineIdx = buffer.indexOf('\n');
        while (newlineIdx !== -1) {
          const line = buffer.slice(0, newlineIdx).trim();
          buffer = buffer.slice(newlineIdx + 1);
          newlineIdx = buffer.indexOf('\n');
          if (!line) continue;
          let msg: StreamMessage;
          try {
            msg = JSON.parse(line) as StreamMessage;
          } catch {
            continue;
          }
          if (msg.type === 'progress') {
            setProgress((prev) => [...prev, msg.event]);
          } else if (msg.type === 'verdict') {
            verdict = VerdictSchema.parse(msg.verdict);
          } else if (msg.type === 'error') {
            throw new Error(msg.error.message);
          }
        }
      }

      if (!verdict) throw new Error('stream closed without a verdict');
      setCurrent(verdict);
      setHistory((prev) => [verdict!, ...prev].slice(0, 20));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <main>
      <h1>Trademark Risk Check</h1>
      <p className="subtitle">
        Pre-filing risk intelligence — ML ranker + agentic policy review across USPTO, domain,
        web, and attorney signals.
      </p>

      <CheckerForm onSubmit={handleSubmit} loading={loading} />

      {error ? <div className="error-banner">{error}</div> : null}

      {current ? <VerdictCard verdict={current} /> : null}

      <AgentTrace trace={current?.trace ?? []} progress={progress} loading={loading} />

      <AuditLog entries={history} />
    </main>
  );
}
