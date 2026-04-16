import { lookupApplicantHistory } from '../src/tools/lookup_applicant_history.js';
import { loadClassifier, type FeatureVector } from '../src/lib/classifier.js';
import { applicantToFeatures, markFeaturesFromPriorArt, DEFAULT_PRE_FILING_FEATURES } from '../src/lib/features.js';

interface Case {
  applicant: string;
  brand: string;
  class?: number;
  expect: 'low' | 'mid' | 'high';
}

const CASES: Case[] = [
  // Blue chips — high volume, real businesses
  { applicant: 'Apple Inc.',                 brand: 'AirTag Pro',      class: 9,  expect: 'low'  },
  { applicant: 'Nike Inc.',                  brand: 'Nike Runners',    class: 25, expect: 'low'  },
  { applicant: 'Microsoft Corporation',      brand: 'WindowsPro',      class: 9,  expect: 'low'  },
  { applicant: 'Google LLC',                 brand: 'AndroidX',        class: 9,  expect: 'low'  },
  // Pro-se burst filers (troll shape)
  { applicant: 'I420 LLC',                   brand: 'LA420',           class: 35, expect: 'high' },
  { applicant: 'Three Hearts Products, LLC', brand: 'ThreeHearts2026', class: 25, expect: 'high' },
  { applicant: 'auburn investments',         brand: 'AuburnCoin',      class: 36, expect: 'high' },
  // Foreign shell LLCs (numeric Ontario corp pattern)
  { applicant: '1660929 Ontario Limited',    brand: 'OntBrand',        class: 35, expect: 'mid'  },
  { applicant: '3681441 CANADA INC.',        brand: 'CanBrand',        class: 35, expect: 'mid'  },
  // Tiny / individual filers
  { applicant: 'Caffe Giorgio',              brand: 'GiorgioBlend',    class: 30, expect: 'low'  },
  // Unknown — neutral priors
  { applicant: 'Nonexistent Holdings ZZZ',   brand: 'FreshBrand',      class: 35, expect: 'low'  },
];

async function main(): Promise<void> {
  const classifier = await loadClassifier();
  console.log(
    `${'applicant'.padEnd(32)} ${'brand'.padEnd(18)} ${'score'.padStart(7)} ${'tier'.padStart(5)} expect`
  );
  console.log('-'.repeat(85));

  for (const c of CASES) {
    const hist = await lookupApplicantHistory({ applicant_name: c.applicant });
    const applicantFeats = hist.ok ? applicantToFeatures(hist.data) : {};
    const markFeats = markFeaturesFromPriorArt(c.brand, c.class);
    const features: FeatureVector = { ...DEFAULT_PRE_FILING_FEATURES, ...applicantFeats, ...markFeats };

    const pred = await classifier.predict(features);

    const line =
      `${c.applicant.padEnd(32)} ${c.brand.padEnd(18)} ${pred.score.toFixed(3).padStart(7)} ` +
      `${pred.tier.padStart(5)} ${c.expect}`;
    console.log(line);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
