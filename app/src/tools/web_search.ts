import {
  WebSearchInputSchema,
  WebSearchOutputSchema,
} from '../schemas/index.js';
import { fetchJson, runTool, ToolError, type FetchImpl, type ToolResult } from '../lib/http.js';
import type { z } from 'zod';

type Input = z.input<typeof WebSearchInputSchema>;
type Output = z.infer<typeof WebSearchOutputSchema>;

const DEFAULT_URL = 'https://google.serper.dev/search';

interface SerperOrganic {
  title?: string;
  link?: string;
  snippet?: string;
}
interface SerperResponse {
  organic?: SerperOrganic[];
}

export async function webSearch(
  input: Input,
  opts: { fetchImpl?: FetchImpl; url?: string; apiKey?: string; timeoutMs?: number } = {}
): Promise<ToolResult<Output>> {
  const parsed = WebSearchInputSchema.parse(input);
  const url = opts.url ?? process.env.SERPER_URL ?? DEFAULT_URL;
  const apiKey = opts.apiKey ?? process.env.SERPER_API_KEY;

  return runTool(async () => {
    if (!apiKey) {
      throw new ToolError('SERPER_API_KEY is not set', 'config');
    }

    const response = await fetchJson<SerperResponse>(url, {
      method: 'POST',
      headers: { 'x-api-key': apiKey },
      body: { q: parsed.query, num: parsed.max_results },
      timeoutMs: opts.timeoutMs ?? 3000,
      ...(opts.fetchImpl ? { fetchImpl: opts.fetchImpl } : {}),
    });

    const organic = response.organic ?? [];
    const results = organic
      .filter((r): r is Required<SerperOrganic> => Boolean(r.title && r.link && r.snippet))
      .map((r) => ({ title: r.title, url: r.link, snippet: r.snippet }));

    const output: Output = { query: parsed.query, results };
    const parseResult = WebSearchOutputSchema.safeParse(output);
    if (!parseResult.success) {
      throw new ToolError(`serper response failed schema validation: ${parseResult.error.message}`, 'parse');
    }
    return parseResult.data;
  });
}
