'use client';

import { useState, type FormEvent } from 'react';
import type { CheckRequest } from '../schemas/index.js';

interface Preset {
  label: string;
  request: CheckRequest;
}

const PRESETS: Preset[] = [
  { label: 'BrewBox Coffee', request: { brand_name: 'BrewBox Coffee', class_code: 30 } },
  { label: 'NovaPay', request: { brand_name: 'NovaPay', class_code: 36 } },
  { label: 'Nike Shoes', request: { brand_name: 'Nike Shoes', class_code: 25 } },
  { label: 'xyzzy troll mark', request: { brand_name: 'xyzzy troll mark' } },
];

interface Props {
  onSubmit: (req: CheckRequest) => void;
  loading: boolean;
}

export function CheckerForm({ onSubmit, loading }: Props) {
  const [brandName, setBrandName] = useState('');
  const [domainName, setDomainName] = useState('');
  const [classCode, setClassCode] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!brandName.trim()) return;
    const req: CheckRequest = {
      brand_name: brandName.trim(),
      ...(domainName.trim() ? { domain_name: domainName.trim() } : {}),
      ...(classCode ? { class_code: Number(classCode) } : {}),
    };
    onSubmit(req);
  };

  const applyPreset = (p: Preset) => {
    setBrandName(p.request.brand_name);
    setDomainName(p.request.domain_name ?? '');
    setClassCode(p.request.class_code ? String(p.request.class_code) : '');
  };

  return (
    <section className="panel">
      <h2>Check a brand</h2>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: 12 }}>
          <label htmlFor="brand">Brand name</label>
          <input
            id="brand"
            type="text"
            value={brandName}
            onChange={(e) => setBrandName(e.target.value)}
            placeholder="e.g. BrewBox Coffee"
            maxLength={200}
            required
            aria-label="brand name"
          />
        </div>
        <div className="row" style={{ marginBottom: 16 }}>
          <div>
            <label htmlFor="domain">Domain (optional)</label>
            <input
              id="domain"
              type="text"
              value={domainName}
              onChange={(e) => setDomainName(e.target.value)}
              placeholder="brewbox.com"
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
              placeholder="30"
            />
          </div>
        </div>
        <button type="submit" disabled={loading || !brandName.trim()}>
          {loading ? 'Checking…' : 'Run risk check'}
        </button>
      </form>
      <div className="presets">
        {PRESETS.map((p) => (
          <button
            type="button"
            key={p.label}
            className="pill"
            onClick={() => applyPreset(p)}
            disabled={loading}
          >
            {p.label}
          </button>
        ))}
      </div>
    </section>
  );
}
