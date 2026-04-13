import {
  CheckAttorneyInputSchema,
  CheckAttorneyOutputSchema,
} from '../schemas/index.js';
import { fetchText, runTool, type FetchImpl, type ToolResult } from '../lib/http.js';
import type { z } from 'zod';

type Input = z.infer<typeof CheckAttorneyInputSchema>;
type Output = z.infer<typeof CheckAttorneyOutputSchema>;

const DEFAULT_URL = 'https://oedci.uspto.gov/OEDCI/practitionerSearch.do';

function stripTags(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function parseBarStatus(text: string): Output['bar_status'] {
  const lower = text.toLowerCase();
  if (/\bsuspended\b/.test(lower)) return 'suspended';
  if (/\binactive\b|\bresigned\b|\bdisbarred\b/.test(lower)) return 'inactive';
  if (/\bactive\b|\bin good standing\b|\bregistered\b/.test(lower)) return 'active';
  return 'unknown';
}

export async function checkAttorney(
  input: Input,
  opts: { fetchImpl?: FetchImpl; url?: string; timeoutMs?: number } = {}
): Promise<ToolResult<Output>> {
  const parsed = CheckAttorneyInputSchema.parse(input);
  const base = opts.url ?? process.env.USPTO_OED_URL ?? DEFAULT_URL;

  return runTool(async () => {
    const url = new URL(base);
    url.searchParams.set('name', parsed.attorney_name);
    if (parsed.bar_number) url.searchParams.set('regNumber', parsed.bar_number);

    const html = await fetchText(url.toString(), {
      timeoutMs: opts.timeoutMs ?? 5000,
      ...(opts.fetchImpl ? { fetchImpl: opts.fetchImpl } : {}),
    });

    const text = stripTags(html);
    const notFound =
      /no results found|no practitioners found|0 results|no matches/i.test(text);

    if (notFound) {
      return CheckAttorneyOutputSchema.parse({
        found: false,
        name: null,
        bar_status: 'unknown',
        disciplinary_history: false,
      });
    }

    const bar_status = parseBarStatus(text);
    const disciplinary_history =
      /\b(disciplined|sanctioned|disciplinary action|censure|reprimand)\b/i.test(text);

    return CheckAttorneyOutputSchema.parse({
      found: true,
      name: parsed.attorney_name,
      bar_status,
      disciplinary_history,
    });
  });
}
