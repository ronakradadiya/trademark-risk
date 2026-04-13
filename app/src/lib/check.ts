import { ZodError } from 'zod';
import {
  CheckRequestSchema,
  type CheckRequest,
  type Verdict,
} from '../schemas/index.js';
import { runCheck, type AgentDeps } from './agent.js';
import type { AuditStore } from './dynamo.js';

export interface HandleCheckDeps extends AgentDeps {
  auditStore?: AuditStore | null;
}

export type HandleCheckResult =
  | { status: 200; verdict: Verdict }
  | { status: 400; error: { message: string; issues?: unknown } }
  | { status: 500; error: { message: string } };

export async function handleCheck(
  body: unknown,
  deps: HandleCheckDeps
): Promise<HandleCheckResult> {
  let req: CheckRequest;
  try {
    req = CheckRequestSchema.parse(body);
  } catch (e) {
    if (e instanceof ZodError) {
      return {
        status: 400,
        error: { message: 'invalid request body', issues: e.issues },
      };
    }
    return { status: 400, error: { message: 'invalid request body' } };
  }

  let verdict: Verdict;
  try {
    verdict = await runCheck(req, deps);
  } catch (e) {
    return {
      status: 500,
      error: { message: `check failed: ${e instanceof Error ? e.message : String(e)}` },
    };
  }

  if (deps.auditStore) {
    // Fire-and-forget-style: audit failures should not fail the response.
    const putResult = await deps.auditStore.put(verdict);
    if (!putResult.ok) {
      console.warn('[audit] failed to persist verdict:', putResult.error);
    }
  }

  return { status: 200, verdict };
}
