'use client';

import type { AgentStep, ProgressEvent } from '../schemas/index.js';

interface Props {
  trace: AgentStep[];
  progress: ProgressEvent[];
  loading: boolean;
}

function statusIcon(status: ProgressEvent['status']): string {
  if (status === 'completed') return '✓';
  if (status === 'failed') return '✗';
  return '⋯';
}

export function AgentTrace({ trace, progress, loading }: Props) {
  const hasProgress = progress.length > 0;
  if (!loading && trace.length === 0 && !hasProgress) return null;

  return (
    <section className="panel">
      <h2>Agent trace</h2>
      {hasProgress ? (
        <div>
          {progress.map((ev, i) => (
            <div key={i} className={`trace-item trace-${ev.kind} status-${ev.status}`}>
              <span className="trace-icon">{statusIcon(ev.status)}</span>
              <span className="trace-kind">{ev.kind}</span>
              <span className="trace-label"> — {ev.label}</span>
              {ev.detail ? <span className="trace-detail"> · {ev.detail}</span> : null}
            </div>
          ))}
          {loading ? <div className="empty">Investigating…</div> : null}
        </div>
      ) : loading ? (
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
