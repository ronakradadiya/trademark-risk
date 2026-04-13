import {
  CheckUsptoMarksInputSchema,
  CheckUsptoMarksOutputSchema,
  type USPTOMark,
} from '../schemas/index.js';
import { fetchJson, runTool, ToolError, type FetchImpl, type ToolResult } from '../lib/http.js';
import type { z } from 'zod';

type Input = z.infer<typeof CheckUsptoMarksInputSchema>;
type Output = z.infer<typeof CheckUsptoMarksOutputSchema>;

const DEFAULT_BASE = 'https://tmsearch.uspto.gov/api/search/tm';

interface UsptoHit {
  serialNumber?: string;
  serial_number?: string;
  markLiteralElement?: string;
  mark?: string;
  ownerName?: string;
  owner?: string;
  filingDate?: string;
  filing_date?: string;
  statusDescription?: string;
  status?: string;
  internationalClass?: Array<string | number> | string;
  classCodes?: Array<string | number>;
}

interface UsptoResponse {
  results?: UsptoHit[];
  hits?: UsptoHit[];
  total?: number;
}

function normalize(hit: UsptoHit): USPTOMark {
  const raw = hit.internationalClass ?? hit.classCodes ?? [];
  const arr = Array.isArray(raw) ? raw : String(raw).split(/[,\s]+/);
  const class_codes = arr
    .map((v) => Number(v))
    .filter((n) => Number.isInteger(n) && n >= 1 && n <= 45);

  return {
    serial_number: String(hit.serialNumber ?? hit.serial_number ?? ''),
    mark_name: String(hit.markLiteralElement ?? hit.mark ?? ''),
    ...(hit.ownerName || hit.owner ? { owner_name: String(hit.ownerName ?? hit.owner) } : {}),
    ...(hit.filingDate || hit.filing_date
      ? { filing_date: String(hit.filingDate ?? hit.filing_date) }
      : {}),
    ...(hit.statusDescription || hit.status
      ? { status: String(hit.statusDescription ?? hit.status) }
      : {}),
    class_codes,
  };
}

export async function checkUsptoMarks(
  input: Input,
  opts: { fetchImpl?: FetchImpl; baseUrl?: string; apiKey?: string; timeoutMs?: number } = {}
): Promise<ToolResult<Output>> {
  const parsed = CheckUsptoMarksInputSchema.parse(input);
  const base = opts.baseUrl ?? process.env.USPTO_TM_SEARCH_URL ?? DEFAULT_BASE;
  const apiKey = opts.apiKey ?? process.env.USPTO_API_KEY;

  return runTool(async () => {
    const url = new URL(base);
    url.searchParams.set('q', parsed.brand_name);
    if (parsed.class_code !== undefined) {
      url.searchParams.set('class', String(parsed.class_code));
    }

    const response = await fetchJson<UsptoResponse>(url.toString(), {
      headers: apiKey ? { 'x-api-key': apiKey } : {},
      timeoutMs: opts.timeoutMs ?? 5000,
      ...(opts.fetchImpl ? { fetchImpl: opts.fetchImpl } : {}),
    });

    const hits = response.results ?? response.hits ?? [];
    const results = hits.map(normalize).filter((m) => m.serial_number && m.mark_name);

    const output: Output = {
      query: parsed.brand_name,
      total: typeof response.total === 'number' ? response.total : results.length,
      results,
    };

    const parseResult = CheckUsptoMarksOutputSchema.safeParse(output);
    if (!parseResult.success) {
      throw new ToolError(`response failed schema validation: ${parseResult.error.message}`, 'parse');
    }
    return parseResult.data;
  });
}
