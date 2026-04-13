import {
  CheckDomainAgeInputSchema,
  CheckDomainAgeOutputSchema,
} from '../schemas/index.js';
import { fetchJson, runTool, ToolError, type FetchImpl, type ToolResult } from '../lib/http.js';
import type { z } from 'zod';

type Input = z.infer<typeof CheckDomainAgeInputSchema>;
type Output = z.infer<typeof CheckDomainAgeOutputSchema>;

const DEFAULT_BASE = 'https://rdap.org/domain/';

interface RdapEvent {
  eventAction: string;
  eventDate: string;
}
interface RdapResponse {
  events?: RdapEvent[];
  errorCode?: number;
}

export async function checkDomainAge(
  input: Input,
  opts: { fetchImpl?: FetchImpl; baseUrl?: string; timeoutMs?: number } = {}
): Promise<ToolResult<Output>> {
  const parsed = CheckDomainAgeInputSchema.parse(input);
  const base = opts.baseUrl ?? process.env.RDAP_BASE_URL ?? DEFAULT_BASE;

  return runTool(async () => {
    const url = `${base}${encodeURIComponent(parsed.domain_name)}`;

    let response: RdapResponse;
    try {
      response = await fetchJson<RdapResponse>(url, {
        timeoutMs: opts.timeoutMs ?? 5000,
        ...(opts.fetchImpl ? { fetchImpl: opts.fetchImpl } : {}),
      });
    } catch (e) {
      if (e instanceof ToolError && e.code === 'http' && e.status === 404) {
        const notFound: Output = {
          domain_name: parsed.domain_name,
          exists: false,
          registered_at: null,
          age_days: null,
        };
        return CheckDomainAgeOutputSchema.parse(notFound);
      }
      throw e;
    }

    const reg = response.events?.find((e) => e.eventAction === 'registration');
    if (!reg) {
      const noReg: Output = {
        domain_name: parsed.domain_name,
        exists: true,
        registered_at: null,
        age_days: null,
      };
      return CheckDomainAgeOutputSchema.parse(noReg);
    }

    const registeredAt = new Date(reg.eventDate);
    if (Number.isNaN(registeredAt.getTime())) {
      throw new ToolError(`invalid registration date: ${reg.eventDate}`, 'parse');
    }
    const ageDays = Math.floor((Date.now() - registeredAt.getTime()) / 86_400_000);

    const output: Output = {
      domain_name: parsed.domain_name,
      exists: true,
      registered_at: registeredAt.toISOString(),
      age_days: ageDays,
    };
    return CheckDomainAgeOutputSchema.parse(output);
  });
}
