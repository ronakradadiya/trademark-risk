'use client';

import type { Verdict } from '../schemas/index.js';
import { POLICIES } from '../lib/policies.js';

interface Props {
  verdict: Verdict;
}

export function VerdictCard({ verdict }: Props) {
  const pct = (v: number) => `${Math.round(v * 100)}%`;

  return (
    <section className="panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 4 }}>Verdict</div>
          <div style={{ fontSize: 20, fontWeight: 600 }}>{verdict.brand}</div>
        </div>
        <span className={`verdict-badge verdict-${verdict.verdict}`}>{verdict.verdict.replace('_', ' ')}</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 20 }}>
        <Metric label="Overall confidence" value={pct(verdict.overall_confidence)} />
        <Metric label="ML risk score" value={pct(verdict.ml.score)} sub={`tier: ${verdict.ml.tier}`} />
        <Metric label="Source" value={verdict.source === 'ml_only' ? 'ML only' : 'ML + agent'} />
      </div>

      <p style={{ color: 'var(--muted)', marginTop: 0, marginBottom: 20 }}>{verdict.summary}</p>

      <h2>Policies</h2>
      <div>
        {(['P1', 'P2', 'P3', 'P4', 'P5'] as const).map((key) => {
          const result = verdict.policies[key];
          const policy = POLICIES[key];
          return (
            <div key={key} className="policy">
              <div>
                <div className="policy-key">{key}</div>
                <div style={{ fontSize: 11, color: 'var(--muted)' }}>
                  {result.triggered ? 'triggered' : 'ok'}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 500 }}>{policy.name}</div>
                <div className="policy-reason">{result.reason}</div>
              </div>
              <div>
                <div
                  style={{
                    fontSize: 12,
                    textAlign: 'right',
                    color: 'var(--muted)',
                    marginBottom: 4,
                  }}
                >
                  {pct(result.confidence)}
                </div>
                <div className="bar">
                  <div
                    className={`bar-fill${result.triggered ? ' triggered' : ''}`}
                    style={{ width: `${Math.max(2, Math.round(result.confidence * 100))}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {label}
      </div>
      <div style={{ fontSize: 18, fontWeight: 600, marginTop: 4 }}>{value}</div>
      {sub ? <div style={{ fontSize: 11, color: 'var(--muted)' }}>{sub}</div> : null}
    </div>
  );
}
