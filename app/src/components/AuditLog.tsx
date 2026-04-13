'use client';

import type { Verdict } from '../schemas/index.js';

interface Props {
  entries: Verdict[];
}

export function AuditLog({ entries }: Props) {
  return (
    <section className="panel">
      <h2>Recent checks</h2>
      {entries.length === 0 ? (
        <div className="empty">No checks yet — run one above.</div>
      ) : (
        <table className="audit-table">
          <thead>
            <tr>
              <th>Applicant</th>
              <th>Brand</th>
              <th>Verdict</th>
              <th>Confidence</th>
              <th>Source</th>
              <th>When</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((v, i) => (
              <tr key={`${v.brand}-${v.checked_at}-${i}`}>
                <td style={{ color: 'var(--muted)' }}>{v.applicant}</td>
                <td>{v.brand}</td>
                <td>
                  <span className={`verdict-badge verdict-${v.verdict}`}>
                    {v.verdict.replace('_', ' ')}
                  </span>
                </td>
                <td>{Math.round(v.overall_confidence * 100)}%</td>
                <td style={{ color: 'var(--muted)' }}>
                  {v.source === 'ml_only' ? 'ML' : 'ML + agent'}
                </td>
                <td style={{ color: 'var(--muted)' }}>{formatTime(v.checked_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return iso;
  }
}
