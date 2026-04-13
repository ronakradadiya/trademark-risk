'use client';

import type { ApplicantHistory, Verdict } from '../schemas/index.js';
import { POLICIES } from '../lib/policies.js';

interface Props {
  verdict: Verdict;
}

export function VerdictCard({ verdict }: Props) {
  const pct = (v: number) => `${Math.round(v * 100)}%`;

  return (
    <section className="panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16, gap: 12 }}>
        <div>
          <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 4 }}>Verdict</div>
          <div style={{ fontSize: 20, fontWeight: 600 }}>{verdict.brand}</div>
          <div style={{ color: 'var(--muted)', fontSize: 13, marginTop: 2 }}>
            filer: {verdict.applicant}
          </div>
        </div>
        <span className={`verdict-badge verdict-${verdict.verdict}`}>{verdict.verdict.replace('_', ' ')}</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 20 }}>
        <Metric label="Overall confidence" value={pct(verdict.overall_confidence)} />
        <Metric label="ML filer risk" value={pct(verdict.ml.score)} sub={`tier: ${verdict.ml.tier}`} />
        <Metric label="Source" value={verdict.source === 'ml_only' ? 'ML only' : 'ML + agent'} />
      </div>

      {verdict.applicant_history ? (
        <ApplicantHistoryCard history={verdict.applicant_history} />
      ) : null}

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

function ApplicantHistoryCard({ history }: { history: ApplicantHistory }) {
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  const sourceLabel =
    history.source === 'fixture'
      ? 'cached USPTO fixture'
      : history.source === 'live'
      ? 'live USPTO'
      : 'not found in USPTO';

  if (!history.found) {
    return (
      <div
        style={{
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: 12,
          marginBottom: 20,
          background: 'var(--panel-2, rgba(255,255,255,0.02))',
        }}
      >
        <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
          Applicant history
        </div>
        <div style={{ fontSize: 13 }}>
          <strong>{history.applicant_name}</strong> — no USPTO history found ({sourceLabel}).
          First-time filer or shell entity; ML defaults to neutral priors.
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        border: '1px solid var(--border)',
        borderRadius: 8,
        padding: 12,
        marginBottom: 20,
        background: 'var(--panel-2, rgba(255,255,255,0.02))',
      }}
    >
      <div style={{ fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
        Applicant history · {sourceLabel}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, fontSize: 13 }}>
        <HistoryStat label="Total filings" value={history.filing_count_total.toLocaleString()} />
        <HistoryStat label="Last 24mo" value={history.filing_count_2yr.toLocaleString()} />
        <HistoryStat label="Abandoned" value={pct(history.abandonment_rate)} danger={history.abandonment_rate > 0.4} />
        <HistoryStat label="Cancelled" value={pct(history.cancellation_rate)} danger={history.cancellation_rate > 0.15} />
        <HistoryStat label="First filing" value={history.first_filing_date ?? '—'} />
        <HistoryStat label="Type" value={history.is_individual ? 'individual' : 'entity'} />
        <HistoryStat label="Foreign" value={history.is_foreign ? 'yes' : 'no'} />
        <HistoryStat
          label="Attorney"
          value={history.attorney_of_record ? history.attorney_of_record : 'pro se'}
          danger={!history.attorney_of_record && history.filing_count_2yr > 5}
        />
      </div>
    </div>
  );
}

function HistoryStat({ label, value, danger }: { label: string; value: string; danger?: boolean }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        {label}
      </div>
      <div
        style={{
          fontSize: 13,
          fontWeight: 500,
          marginTop: 2,
          color: danger ? 'var(--high, #f87171)' : 'inherit',
        }}
      >
        {value}
      </div>
    </div>
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
