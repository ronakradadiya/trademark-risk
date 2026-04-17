'use client';

import { useState, type FormEvent } from 'react';
import type { CheckRequest } from '../schemas/index.js';

interface Preset {
  label: string;
  tagline: string;
  request: CheckRequest;
}

const PRESETS: Preset[] = [
  {
    label: 'Apple + AirTag Pro',
    tagline: 'blue-chip filer, clean',
    request: {
      brand_name: 'AirTag Pro',
      applicant_name: 'Apple Inc.',
      class_code: 9,
    },
  },
  {
    label: 'Safetykit + XBox',
    tagline: 'clean filer, famous-mark collision',
    request: {
      brand_name: 'XBox',
      applicant_name: 'Safetykit Inc.',
    },
  },
  {
    label: 'Leo Stoller + EZ Profits',
    tagline: 'famous troll filer',
    request: {
      brand_name: 'EZ Profits',
      applicant_name: 'Leo Stoller',
      class_code: 35,
    },
  },
  {
    label: 'I420 LLC + LA420',
    tagline: 'real shell-LLC filer',
    request: {
      brand_name: 'LA420',
      applicant_name: 'I420 LLC',
      class_code: 35,
    },
  },
  {
    label: 'Nike + Nike Runners',
    tagline: 'established owner, same class',
    request: {
      brand_name: 'Nike Runners',
      applicant_name: 'Nike Inc.',
      class_code: 25,
    },
  },
];

interface Props {
  onSubmit: (req: CheckRequest) => void;
  loading: boolean;
}

export function CheckerForm({ onSubmit, loading }: Props) {
  const [brandName, setBrandName] = useState('');
  const [applicantName, setApplicantName] = useState('');
  const [domainName, setDomainName] = useState('');
  const [classCode, setClassCode] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!brandName.trim() || !applicantName.trim()) return;
    const req: CheckRequest = {
      brand_name: brandName.trim(),
      applicant_name: applicantName.trim(),
      ...(domainName.trim() ? { domain_name: domainName.trim() } : {}),
      ...(classCode ? { class_code: Number(classCode) } : {}),
    };
    onSubmit(req);
  };

  const applyPreset = (p: Preset) => {
    setBrandName(p.request.brand_name);
    setApplicantName(p.request.applicant_name);
    setDomainName(p.request.domain_name ?? '');
    setClassCode(p.request.class_code ? String(p.request.class_code) : '');
  };

  return (
    <section className="panel">
      <h2>Due-diligence check</h2>
      <form onSubmit={handleSubmit}>
        <div className="row" style={{ marginBottom: 12 }}>
          <div>
            <label htmlFor="applicant">Applicant / filer</label>
            <input
              id="applicant"
              type="text"
              value={applicantName}
              onChange={(e) => setApplicantName(e.target.value)}
              placeholder="e.g. I420 LLC"
              maxLength={200}
              required
              aria-label="applicant name"
            />
          </div>
          <div>
            <label htmlFor="brand">Brand they want to file</label>
            <input
              id="brand"
              type="text"
              value={brandName}
              onChange={(e) => setBrandName(e.target.value)}
              placeholder="e.g. LA420"
              maxLength={200}
              required
              aria-label="brand name"
            />
          </div>
        </div>
        <div className="row" style={{ marginBottom: 16 }}>
          <div>
            <label htmlFor="domain">Domain (optional)</label>
            <input
              id="domain"
              type="text"
              value={domainName}
              onChange={(e) => setDomainName(e.target.value)}
              placeholder="example.com"
            />
          </div>
          <div>
            <label htmlFor="class">Nice class (optional)</label>
            <input
              id="class"
              type="number"
              min={1}
              max={45}
              value={classCode}
              onChange={(e) => setClassCode(e.target.value)}
              placeholder="35"
            />
          </div>
        </div>
        <button
          type="submit"
          disabled={loading || !brandName.trim() || !applicantName.trim()}
        >
          {loading ? 'Investigating…' : 'Run due-diligence check'}
        </button>
      </form>
      <div className="presets">
        <div
          style={{
            color: 'var(--muted)',
            fontSize: 12,
            marginBottom: 6,
            width: '100%',
          }}
        >
          Demo scenarios
        </div>
        {PRESETS.map((p) => (
          <button
            type="button"
            key={p.label}
            className="pill"
            onClick={() => applyPreset(p)}
            disabled={loading}
            title={p.tagline}
          >
            {p.label}
          </button>
        ))}
      </div>
    </section>
  );
}
