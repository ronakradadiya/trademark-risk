import { checkUsptoMarks } from '../src/tools/check_uspto_marks.js';
import Database from 'better-sqlite3';

async function main(): Promise<void> {
  const header = (s: string): void => console.log(`\n=== ${s} ===`);

  header('check_uspto_marks: HeyPoco (no class)');
  const any = await checkUsptoMarks({ brand_name: 'HeyPoco' });
  console.log(any);

  header('check_uspto_marks: Hey Poco (two-word)');
  const two = await checkUsptoMarks({ brand_name: 'Hey Poco' });
  console.log(two);

  header('direct SQLite: any mark containing "poco"');
  const db = new Database('/Users/ronakradadiya/Documents/projects/trademark-risk/data/uspto.sqlite', { readonly: true });
  const rows = db
    .prepare(
      `SELECT m.serial, m.mark, m.classes, m.filing_date, m.status_code, m.owner_name
       FROM marks_fts f JOIN marks m ON f.rowid = m.rowid
       WHERE marks_fts MATCH '"poco"'
         AND (m.status_code IS NULL OR m.status_code < '600' OR (m.status_code >= '800' AND m.status_code < '900'))
       ORDER BY bm25(marks_fts) LIMIT 20`
    )
    .all();
  console.log('live poco-containing marks:', rows.length);
  console.log(rows);

  header('direct SQLite: mark_norm LIKE "%heypoco%"');
  const hp = db
    .prepare(`SELECT serial, mark, status_code, owner_name FROM marks WHERE mark_norm LIKE '%heypoco%'`)
    .all();
  console.log(hp);

  header('direct SQLite: mark_norm LIKE "hey poco"');
  const hpSpace = db
    .prepare(`SELECT serial, mark, status_code, owner_name FROM marks WHERE mark_norm LIKE '%hey poco%'`)
    .all();
  console.log(hpSpace);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
