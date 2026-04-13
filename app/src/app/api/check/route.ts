import { NextResponse } from 'next/server';
import { handleCheck } from '../../../lib/check.js';
import { loadClassifier, type Classifier } from '../../../lib/classifier.js';
import { createAuditStore, type AuditStore } from '../../../lib/dynamo.js';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

let classifierPromise: Promise<Classifier> | null = null;
let auditStore: AuditStore | null = null;

function getClassifier(): Promise<Classifier> {
  if (!classifierPromise) {
    classifierPromise = loadClassifier();
  }
  return classifierPromise;
}

function getAuditStore(): AuditStore | null {
  if (process.env.DYNAMO_AUDIT_DISABLED === '1') return null;
  if (!auditStore) {
    try {
      auditStore = createAuditStore();
    } catch (e) {
      console.warn('[audit] could not initialize store:', e);
      auditStore = null;
    }
  }
  return auditStore;
}

export async function POST(request: Request): Promise<NextResponse> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: { message: 'request body must be valid JSON' } },
      { status: 400 }
    );
  }

  const classifier = await getClassifier();
  const result = await handleCheck(body, {
    classifier,
    auditStore: getAuditStore(),
  });

  if (result.status === 200) {
    return NextResponse.json(result.verdict, { status: 200 });
  }
  return NextResponse.json({ error: result.error }, { status: result.status });
}
