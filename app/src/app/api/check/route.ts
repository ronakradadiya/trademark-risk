import { NextResponse } from 'next/server';
import { handleCheck } from '../../../lib/check.js';
import { loadClassifier, type Classifier } from '../../../lib/classifier.js';
import { createAuditStore, type AuditStore } from '../../../lib/dynamo.js';
import type { ProgressEvent } from '../../../schemas/index.js';

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

export async function POST(request: Request): Promise<Response> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: { message: 'request body must be valid JSON' } },
      { status: 400 }
    );
  }

  const wantsStream = (request.headers.get('accept') ?? '').includes('application/x-ndjson');
  const classifier = await getClassifier();

  if (!wantsStream) {
    const result = await handleCheck(body, {
      classifier,
      auditStore: getAuditStore(),
    });
    if (result.status === 200) {
      return NextResponse.json(result.verdict, { status: 200 });
    }
    return NextResponse.json({ error: result.error }, { status: result.status });
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const write = (obj: unknown) => {
        controller.enqueue(encoder.encode(JSON.stringify(obj) + '\n'));
      };
      const onProgress = (event: ProgressEvent) => {
        write({ type: 'progress', event });
      };
      try {
        const result = await handleCheck(body, {
          classifier,
          auditStore: getAuditStore(),
          onProgress,
        });
        if (result.status === 200) {
          write({ type: 'verdict', verdict: result.verdict });
        } else {
          write({ type: 'error', status: result.status, error: result.error });
        }
      } catch (e) {
        write({
          type: 'error',
          status: 500,
          error: { message: e instanceof Error ? e.message : String(e) },
        });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'content-type': 'application/x-ndjson; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      'x-accel-buffering': 'no',
    },
  });
}
