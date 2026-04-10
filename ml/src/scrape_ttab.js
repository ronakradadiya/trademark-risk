/**
 * Downloads TTAB XML data and extracts cancellation serial numbers.
 *
 * Approach:
 * 1. Download available TTAB daily XML files from Archive.org (2011 data confirmed available)
 * 2. Parse XML to extract cancellation proceedings with serial numbers
 * 3. Combine with CANG/PETC events from event.csv for full coverage
 * 4. Save consolidated fraud serial numbers list
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');
const { execSync } = require('child_process');

const TTAB_DIR = path.join(__dirname, '..', 'data', 'raw', 'ttab');

function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);
    const client = url.startsWith('https') ? https : http;
    const req = client.get(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (research project)' },
      timeout: 30000,
    }, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        file.close();
        try { fs.unlinkSync(destPath); } catch(e) {}
        return downloadFile(response.headers.location, destPath).then(resolve).catch(reject);
      }
      if (response.statusCode !== 200) {
        file.close();
        try { fs.unlinkSync(destPath); } catch(e) {}
        return reject(new Error(`HTTP ${response.statusCode} for ${url}`));
      }
      response.pipe(file);
      file.on('finish', () => { file.close(); resolve(destPath); });
    });
    req.on('error', (err) => {
      file.close();
      try { fs.unlinkSync(destPath); } catch(e) {}
      reject(err);
    });
    req.on('timeout', () => {
      req.destroy();
      reject(new Error(`Timeout downloading ${url}`));
    });
  });
}

/**
 * Download TTAB daily XML files from Archive.org.
 * Available: https://archive.org/download/uspto-ttab-2011/tt11MMDD.zip
 */
async function downloadFromArchiveOrg() {
  console.log('=== Downloading TTAB XML from Archive.org (2011) ===\n');

  const zipDir = path.join(TTAB_DIR, 'zips');
  const xmlDir = path.join(TTAB_DIR, 'xml');
  fs.mkdirSync(zipDir, { recursive: true });
  fs.mkdirSync(xmlDir, { recursive: true });

  // Generate all daily file names for 2011 (Jan-May based on Archive.org listing)
  const dates = [];
  for (let month = 1; month <= 5; month++) {
    const daysInMonth = new Date(2011, month, 0).getDate();
    for (let day = 1; day <= daysInMonth; day++) {
      const mm = String(month).padStart(2, '0');
      const dd = String(day).padStart(2, '0');
      dates.push(`11${mm}${dd}`);
    }
  }

  let downloaded = 0;
  let failed = 0;
  const batchSize = 5;

  for (let i = 0; i < dates.length; i += batchSize) {
    const batch = dates.slice(i, i + batchSize);
    const promises = batch.map(async (dateStr) => {
      const zipName = `tt${dateStr}.zip`;
      const zipPath = path.join(zipDir, zipName);
      const url = `https://archive.org/download/uspto-ttab-2011/${zipName}`;

      if (fs.existsSync(zipPath) && fs.statSync(zipPath).size > 100) {
        downloaded++;
        return;
      }

      try {
        await downloadFile(url, zipPath);
        downloaded++;
      } catch (e) {
        failed++;
        // Weekend/holiday — no filing that day, expected
      }
    });

    await Promise.all(promises);
    process.stdout.write(`\r  Downloaded: ${downloaded}, Failed: ${failed} of ${dates.length}`);
  }

  console.log(`\n  Total downloaded: ${downloaded} files`);

  // Unzip all files
  console.log('\nExtracting ZIP files...');
  const zipFiles = fs.readdirSync(zipDir).filter(f => f.endsWith('.zip'));
  let extracted = 0;

  for (const zipFile of zipFiles) {
    const zipPath = path.join(zipDir, zipFile);
    try {
      // Check if it's actually a zip file (not HTML error page)
      const header = fs.readFileSync(zipPath, { encoding: null }).slice(0, 4);
      if (header[0] !== 0x50 || header[1] !== 0x4B) {
        // Not a ZIP file (probably 404 HTML)
        fs.unlinkSync(zipPath);
        continue;
      }
      execSync(`unzip -o -q "${zipPath}" -d "${xmlDir}"`, { timeout: 10000 });
      extracted++;
    } catch (e) {
      // corrupt or not a zip
    }
  }

  console.log(`  Extracted: ${extracted} XML files`);
  return xmlDir;
}

/**
 * Parse TTAB XML files to extract cancellation proceedings and serial numbers.
 */
function parseTTABXml(xmlDir) {
  console.log('\n=== Parsing TTAB XML files ===\n');

  const xmlFiles = fs.readdirSync(xmlDir).filter(f => f.endsWith('.xml'));
  console.log(`Found ${xmlFiles.length} XML files to parse`);

  const proceedings = {
    CAN: { count: 0, serials: new Set(), statusCounts: {} },  // Cancellation
    OPP: { count: 0, serials: new Set(), statusCounts: {} },  // Opposition
    EXA: { count: 0, serials: new Set(), statusCounts: {} },  // Ex Parte Appeal
    EXT: { count: 0, serials: new Set(), statusCounts: {} },  // Extension
    OTHER: { count: 0, serials: new Set(), statusCounts: {} },
  };

  for (const xmlFile of xmlFiles) {
    const xmlPath = path.join(xmlDir, xmlFile);
    const content = fs.readFileSync(xmlPath, 'utf-8');

    // Simple regex-based parsing (faster than DOM for this structure)
    const entryRegex = /<proceeding-entry>([\s\S]*?)<\/proceeding-entry>/g;
    let match;

    while ((match = entryRegex.exec(content)) !== null) {
      const entry = match[1];

      // Extract type-code
      const typeMatch = entry.match(/<type-code>(\w+)<\/type-code>/);
      const typeCode = typeMatch ? typeMatch[1] : 'OTHER';

      // Extract status-code
      const statusMatch = entry.match(/<status-code>(\d+)<\/status-code>/);
      const statusCode = statusMatch ? statusMatch[1] : 'unknown';

      // Extract defendant serial numbers (role-code = D)
      const partyRegex = /<party>([\s\S]*?)<\/party>/g;
      let partyMatch;
      while ((partyMatch = partyRegex.exec(entry)) !== null) {
        const party = partyMatch[1];
        const roleMatch = party.match(/<role-code>(\w)<\/role-code>/);
        if (roleMatch && roleMatch[1] === 'D') {
          // This is a defendant — extract serial numbers
          const serialRegex = /<serial-number>(\d+)<\/serial-number>/g;
          let serialMatch;
          while ((serialMatch = serialRegex.exec(party)) !== null) {
            const serial = parseInt(serialMatch[1], 10);
            if (serial > 0) {
              const bucket = proceedings[typeCode] || proceedings.OTHER;
              bucket.serials.add(serial);
              bucket.count++;
              bucket.statusCounts[statusCode] = (bucket.statusCounts[statusCode] || 0) + 1;
            }
          }
        }
      }
    }
  }

  // Summary
  console.log('\n=== TTAB Proceedings Summary ===\n');
  for (const [type, data] of Object.entries(proceedings)) {
    if (data.count > 0) {
      const typeName = { CAN: 'Cancellation', OPP: 'Opposition', EXA: 'Ex Parte Appeal', EXT: 'Extension', OTHER: 'Other' }[type] || type;
      console.log(`${typeName} (${type}):`);
      console.log(`  Proceedings: ${data.count}`);
      console.log(`  Unique defendant serials: ${data.serials.size}`);
      console.log(`  Status codes: ${JSON.stringify(data.statusCounts)}`);
    }
  }

  return proceedings;
}

async function main() {
  fs.mkdirSync(TTAB_DIR, { recursive: true });

  console.log('TTAB Data Scraper\n');
  console.log('Note: ODP WAF blocks programmatic access to the bulk data API.');
  console.log('Using Archive.org mirror (2011 data) + event.csv CANG codes for full coverage.\n');

  // Step 1: Download from Archive.org
  const xmlDir = await downloadFromArchiveOrg();

  // Step 2: Parse XML files
  const proceedings = parseTTABXml(xmlDir);

  // Step 3: Save extracted serial numbers
  const cancellationSerials = [...proceedings.CAN.serials];
  const oppositionSerials = [...proceedings.OPP.serials];

  const output = {
    source: 'Archive.org USPTO TTAB 2011 (daily XML files)',
    extractedAt: new Date().toISOString(),
    cancellation: {
      count: cancellationSerials.length,
      serials: cancellationSerials,
    },
    opposition: {
      count: oppositionSerials.length,
      serials: oppositionSerials,
    },
  };

  const outputPath = path.join(TTAB_DIR, 'ttab_serials.json');
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\nSaved ${cancellationSerials.length} cancellation + ${oppositionSerials.length} opposition serials to ${outputPath}`);

  // Step 4: Show how many overlap with our case_file.csv
  console.log('\n=== Next Steps ===');
  console.log('Run the Python script to:');
  console.log('1. Load ttab_serials.json');
  console.log('2. Cross-reference with case_file.csv');
  console.log('3. Combine with CANG events from event.csv');
  console.log('4. Create improved fraud labels');
}

main().catch(console.error);
