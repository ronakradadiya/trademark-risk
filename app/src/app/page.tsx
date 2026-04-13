'use client';

import { useState } from 'react';
import { CheckerForm } from '../components/CheckerForm.js';
import { VerdictCard } from '../components/VerdictCard.js';
import { AgentTrace } from '../components/AgentTrace.js';
import { AuditLog } from '../components/AuditLog.js';
import { VerdictSchema, type CheckRequest, type Verdict } from '../schemas/index.js';

export default function Page() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [current, setCurrent] = useState<Verdict | null>(null);
  const [history, setHistory] = useState<Verdict[]>([]);

  const handleSubmit = async (req: CheckRequest) => {
    setLoading(true);
    setError(null);
    setCurrent(null);
    try {
      const res = await fetch('/api/check', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(req),
      });
      const json = (await res.json()) as unknown;
      if (!res.ok) {
        const err = json as { error?: { message?: string } };
        throw new Error(err.error?.message ?? `request failed (${res.status})`);
      }
      const verdict = VerdictSchema.parse(json);
      setCurrent(verdict);
      setHistory((prev) => [verdict, ...prev].slice(0, 20));
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

      <AgentTrace trace={current?.trace ?? []} loading={loading} />

      <AuditLog entries={history} />
    </main>
  );
}
