export type FetchImpl = typeof fetch;

export class ToolError extends Error {
  constructor(
    message: string,
    public readonly code: 'timeout' | 'network' | 'http' | 'parse' | 'config',
    public readonly status?: number
  ) {
    super(message);
    this.name = 'ToolError';
  }
}

export interface FetchJsonOptions {
  method?: 'GET' | 'POST';
  headers?: Record<string, string>;
  body?: unknown;
  timeoutMs?: number;
  fetchImpl?: FetchImpl;
}

export async function fetchJson<T = unknown>(
  url: string,
  opts: FetchJsonOptions = {}
): Promise<T> {
  const { method = 'GET', headers = {}, body, timeoutMs = 5000, fetchImpl = fetch } = opts;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let res: Response;
  try {
    res = await fetchImpl(url, {
      method,
      headers: {
        accept: 'application/json',
        ...(body !== undefined ? { 'content-type': 'application/json' } : {}),
        ...headers,
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (e) {
    if (e instanceof Error && e.name === 'AbortError') {
      throw new ToolError(`request timed out after ${timeoutMs}ms`, 'timeout');
    }
    throw new ToolError(
      `network failure: ${e instanceof Error ? e.message : String(e)}`,
      'network'
    );
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    throw new ToolError(`HTTP ${res.status} from ${url}`, 'http', res.status);
  }

  try {
    return (await res.json()) as T;
  } catch (e) {
    throw new ToolError(
      `failed to parse JSON: ${e instanceof Error ? e.message : String(e)}`,
      'parse'
    );
  }
}

export async function fetchText(
  url: string,
  opts: Omit<FetchJsonOptions, 'body'> = {}
): Promise<string> {
  const { method = 'GET', headers = {}, timeoutMs = 5000, fetchImpl = fetch } = opts;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetchImpl(url, {
      method,
      headers: { accept: 'text/html,application/xhtml+xml', ...headers },
      signal: controller.signal,
    });
  } catch (e) {
    if (e instanceof Error && e.name === 'AbortError') {
      throw new ToolError(`request timed out after ${timeoutMs}ms`, 'timeout');
    }
    throw new ToolError(
      `network failure: ${e instanceof Error ? e.message : String(e)}`,
      'network'
    );
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) throw new ToolError(`HTTP ${res.status} from ${url}`, 'http', res.status);
  return res.text();
}

export type ToolResult<T> =
  | { ok: true; data: T; latency_ms: number }
  | { ok: false; error: { code: ToolError['code']; message: string; status?: number }; latency_ms: number };

export async function runTool<T>(fn: () => Promise<T>): Promise<ToolResult<T>> {
  const start = Date.now();
  try {
    const data = await fn();
    return { ok: true, data, latency_ms: Date.now() - start };
  } catch (e) {
    const latency_ms = Date.now() - start;
    if (e instanceof ToolError) {
      return {
        ok: false,
        error: { code: e.code, message: e.message, ...(e.status ? { status: e.status } : {}) },
        latency_ms,
      };
    }
    return {
      ok: false,
      error: {
        code: 'network',
        message: e instanceof Error ? e.message : String(e),
      },
      latency_ms,
    };
  }
}
