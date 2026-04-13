'use client';

import type { AgentStep } from '../schemas/index.js';

interface Props {
  trace: AgentStep[];
  loading: boolean;
}

export function AgentTrace({ trace, loading }: Props) {
  if (!loading && trace.length === 0) return null;

  return (
    <section className="panel">
      <h2>Agent trace</h2>
      {loading && trace.length === 0 ? (
        <div className="empty">Investigating…</div>
      ) : (
        <div>
          {trace.map((step, i) => (
            <div key={i} className="trace-item">
              <span>[{step.step}] </span>
              <span className="tool">{step.tool}</span>
              <span> — {step.duration_ms}ms</span>
              {step.error ? <span className="error"> · error: {step.error}</span> : null}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
