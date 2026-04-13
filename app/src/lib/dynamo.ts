import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import {
  DynamoDBDocumentClient,
  PutCommand,
  GetCommand,
  QueryCommand,
  ScanCommand,
} from '@aws-sdk/lib-dynamodb';
import { randomUUID } from 'node:crypto';
import {
  AuditRecordSchema,
  type AuditRecord,
  type Verdict,
} from '../schemas/index.js';

export const TABLE_NAME = process.env.DYNAMO_TABLE ?? 'trademark-risk-checks';
export const TTL_SECONDS = 90 * 24 * 60 * 60;

export class DynamoError extends Error {
  constructor(message: string, public readonly code: 'write' | 'read' | 'list') {
    super(message);
    this.name = 'DynamoError';
  }
}

export interface AuditStore {
  put(verdict: Verdict): Promise<{ ok: true; record: AuditRecord } | { ok: false; error: string }>;
  get(id: string, checked_at: string): Promise<AuditRecord | null>;
  listRecent(limit?: number): Promise<AuditRecord[]>;
}

function defaultDocClient(): DynamoDBDocumentClient {
  const client = new DynamoDBClient({});
  return DynamoDBDocumentClient.from(client, {
    marshallOptions: { removeUndefinedValues: true },
  });
}

export function createAuditStore(
  doc: DynamoDBDocumentClient = defaultDocClient(),
  tableName: string = TABLE_NAME
): AuditStore {
  return {
    async put(verdict) {
      const id = randomUUID();
      const ttl = Math.floor(Date.now() / 1000) + TTL_SECONDS;
      const record: AuditRecord = AuditRecordSchema.parse({ ...verdict, id, ttl });
      try {
        await doc.send(new PutCommand({ TableName: tableName, Item: record }));
        return { ok: true, record };
      } catch (e) {
        return {
          ok: false,
          error: e instanceof Error ? e.message : String(e),
        };
      }
    },

    async get(id, checked_at) {
      try {
        const res = await doc.send(
          new GetCommand({ TableName: tableName, Key: { id, checked_at } })
        );
        if (!res.Item) return null;
        const parsed = AuditRecordSchema.safeParse(res.Item);
        return parsed.success ? parsed.data : null;
      } catch (e) {
        throw new DynamoError(
          `get failed: ${e instanceof Error ? e.message : String(e)}`,
          'read'
        );
      }
    },

    async listRecent(limit = 20) {
      try {
        const res = await doc.send(
          new ScanCommand({ TableName: tableName, Limit: limit })
        );
        const items = res.Items ?? [];
        return items
          .map((item) => AuditRecordSchema.safeParse(item))
          .filter((r): r is { success: true; data: AuditRecord } => r.success)
          .map((r) => r.data);
      } catch (e) {
        throw new DynamoError(
          `list failed: ${e instanceof Error ? e.message : String(e)}`,
          'list'
        );
      }
    },
  };
}

// Re-export commands to allow tests to construct a fake DynamoDBDocumentClient
// by supplying a `send` method.
export { PutCommand, GetCommand, QueryCommand, ScanCommand };
