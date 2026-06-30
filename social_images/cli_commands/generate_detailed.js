// Detailed images for 6 priority commands. Output verified against
// crates/cjc-cli/src/commands/*.rs source. Layout: terminal left + flags right.

const puppeteer = require('puppeteer');
const path = require('path');

const COMMANDS = {
  doctor: {
    cat: 'Inspection & Diagnostics',
    accent: '#bc8cff',
    desc: 'Project-wide health scan — parse errors, schema drift, nondeterminism, file rot',
    cmd: 'cjcl doctor . --strict --report findings.json',
    output: [
      '[doctor] Diagnosing `.`…',
      '',
      'Category          Count',
      '────────────────  ─────',
      'CJC source files  12',
      'Snap files        3',
      'CSV/TSV files     45',
      'Parquet files     8',
      'Total files       200',
      '',
      'ERROR  cache.snap   — corrupt or unreadable snap file',
      'WARN   sales.csv    — 3 ragged row(s)',
      'INFO   train.cjcl   — HashMap may have nondeterministic iteration',
      '',
      'Errors 1  ·  Warnings 2  ·  Info 3',
      'ISSUES FOUND',
    ],
    what: 'Walks a project tree and runs every file through the right validator: CJC sources are parsed, snaps are decoded, CSVs are checked for ragged rows, and source code is grep\u2011scanned for nondeterminism patterns (HashMap, rand(), time()).',
    flags: [
      ['--fix',           'auto-fix safe issues (whitespace, ragged rows)'],
      ['--dry-run',       'preview fixes without writing'],
      ['--strict',        'exit 1 on warnings, not just errors'],
      ['--category <k>',  'filter: parse · csv · snap · determinism · …'],
      ['--report <file>', 'write full findings as JSON'],
    ],
    note: 'one command, full project audit · CI-ready exit codes',
  },

  drift: {
    cat: 'Data & Pipeline',
    accent: '#58a6ff',
    desc: 'Mathematical file diff — per-cell tolerance, NaN tracking, Frobenius norm',
    cmd: 'cjcl drift baseline.csv current.csv --tolerance 1e-4',
    output: [
      'Metric             Value',
      '─────────────────  ────────',
      'Rows (A)           5000',
      'Rows (B)           5000',
      'Schema match       true',
      'Cell differences   42',
      'NaN divergences    3',
      'Max deviation      0.000158',
      'Mean deviation     0.000041',
      'Frobenius norm     0.001824',
      '',
      'First differences:',
      '  [1,2]: "42.5"  ->  "42.501234"',
      '  [5,7]: "3.14"  ->  "3.14159"',
      '',
      'identical: false',
    ],
    what: 'Picks the right comparison engine for the file type (CSV cells, JSONL records, plain text lines, or numeric tensors) and reports both qualitative diffs and quantitative norms — so you can answer "is this drift real or just floating-point noise?"',
    flags: [
      ['--tolerance <e>', 'numeric tolerance (default 0.0)'],
      ['--max-diffs <N>', 'cap shown differences (default 50)'],
      ['--csv | --jsonl', 'force comparison mode'],
      ['--fail-on-diff',  'exit 1 if any difference found'],
      ['--report <file>', 'write full diff report as JSON'],
    ],
    note: 'CSV · JSONL · text · numeric · Frobenius norm · per-cell',
  },

  flow: {
    cat: 'Data & Pipeline',
    accent: '#58a6ff',
    desc: 'Streaming aggregation over CSV/TSV/JSONL — single pass, O(ncols) memory, Kahan summation',
    cmd: 'cjcl flow sales.csv --op sum,mean,min,max,std --verify',
    output: [
      '(5000 rows processed)',
      '',
      'Column    sum (Kahan)   mean    min    max       count  std',
      '────────  ────────────  ──────  ─────  ────────  ─────  ──────',
      'revenue   1234567.89    246.91  10.50  9999.99   5000   90.73',
      'quantity  24500.00      4.90    1.00   100.00    5000   12.52',
      'price     234567.80     46.91   0.50   1999.99   5000   111.11',
      '',
      'verify: PASS — both runs produced identical output',
    ],
    what: 'Reads the file once, keeps one accumulator per column, and emits a tidy table. Sums use Kahan compensated summation so floating-point drift can\u2019t silently corrupt totals. --verify runs the whole pipeline twice and confirms the output bytes match.',
    flags: [
      ['--op sum,mean,…',  'comma-separated aggregates'],
      ['--columns a,b,c',  'restrict to named columns'],
      ['--top <N>',        'show only top N columns'],
      ['--verify',         'run twice, compare output (determinism gate)'],
      ['--out <file>',     'write table to file instead of stdout'],
    ],
    note: 'single pass · Kahan compensated · O(ncols) memory',
  },

  schema: {
    cat: 'Inspection & Diagnostics',
    accent: '#bc8cff',
    desc: 'Type inference + schema validation — null counts, uniques, min/max, save & diff',
    cmd: 'cjcl schema data.csv --full --save schema.json',
    output: [
      'Schema for `data.csv` (5000 rows sampled):',
      '',
      'Column    Type    Nulls  Unique  Min          Max',
      '────────  ──────  ─────  ──────  ───────────  ───────────',
      'id        int     0      5000    1            5000',
      'name      string  45     4800    —            —',
      'revenue   float   12     4988    0.50         9999.99',
      'active    bool    0      2       —            —',
      'created   mixed   8      4820    2024-01-01   2025-04-25',
      '',
      'saved → schema.json',
    ],
    what: 'Infers a column\u2011by\u2011column schema (type, null count, unique count, min/max) by sampling rows. Save it once, then run --check on every new file to gate CI on schema drift, or --diff to see exactly which columns changed.',
    flags: [
      ['--full',          'include type-distribution % + samples'],
      ['--save <f.json>', 'persist inferred schema for reuse'],
      ['--check <f.json>','validate file against saved schema (exit 1)'],
      ['--diff <f.json>', 'show schema delta vs saved baseline'],
      ['--strict',        'treat type mismatches as errors'],
    ],
    note: 'CSV · TSV · JSONL · save once · check forever',
  },

  forge: {
    cat: 'Data & Pipeline',
    accent: '#58a6ff',
    desc: 'Content-addressable pipeline runner — every output gets a SHA-256 identity',
    cmd: 'cjcl forge run pipeline.cjcl --seed 42',
    output: [
      'Property       Value',
      '─────────────  ──────────────────────────────────',
      'Source         pipeline.cjcl',
      'Seed           42',
      'Output lines   127',
      'Output bytes   4356',
      'SHA-256        a3f8d9e2c1b4a5f6e7d8c9b0a1f2e3d4',
      'Artifact       .cjc-forge/a3f8d9e2c1b4….artifact',
      '',
      'forged',
      '',
      '$ cjcl forge verify pipeline.cjcl a3f8d9e2c1b4',
      'verified',
    ],
    what: 'Runs your script with a fixed seed, hashes the output, and stores it in a local content-addressable cache. Re\u2011running with the same inputs hits the cache; re-running with drift produces a different hash — turning "did the result change?" into a one-line check.',
    flags: [
      ['run <file>',          'execute & cache output by SHA-256'],
      ['verify <file> <hash>','re-run and confirm hash match'],
      ['list',                'show all cached artifacts'],
      ['show <hash-prefix>',  'print cached artifact content'],
      ['--seed <N>',          'RNG seed (default 42)'],
    ],
    note: 'reproducibility as a primitive · cache in .cjc-forge/',
  },

  patch: {
    cat: 'Data & Pipeline',
    accent: '#58a6ff',
    desc: 'Streaming, type-aware data transforms — NaN fill, impute, replace, drop, rename',
    cmd: 'cjcl patch sales.csv --nan-fill 0 --drop internal_id --plan',
    output: [
      'Patch Plan',
      '  source:    sales.csv',
      '  format:    CSV',
      '  transforms (3):',
      '    1. nan-fill: replace NaN/NA/null/None with "0" in all columns',
      '    2. replace:  column "status", "pending" -> "completed"',
      '    3. drop:     remove column "internal_id"',
      '',
      '$ cjcl patch sales.csv --nan-fill 0 --drop internal_id --in-place --backup',
      'patched 5000 rows',
      'backup → sales.csv.bak',
    ],
    what: 'A streaming editor for tabular data. Applies a stack of column-aware transforms in one pass with O(ncols) memory. --plan prints the structured transform list before touching anything; --in-place writes via a temp file + atomic rename so a crash never leaves a half-written CSV.',
    flags: [
      ['--nan-fill <v>',     'replace NaN/NA/null in numeric cols'],
      ['--impute <col>',     'fill NaN with column mean (two-pass)'],
      ['--replace <c> <a> <b>', 'rewrite exact values in column'],
      ['--drop <col> / --rename', 'column-level edits'],
      ['--plan / --dry-run', 'preview transforms before writing'],
    ],
    note: 'streaming · atomic --in-place · --backup · --plan',
  },
};

function buildHtml(name, c) {
  const cmdHtml = c.cmd
    .replace(/cjcl/g, '<span style="color:#3fb950">cjcl</span>')
    .replace(/(--[a-z-]+)/g, '<span style="color:#79c0ff">$1</span>')
    .replace(/('[^']*'|"[^"]*")/g, '<span style="color:#a5d6ff">$1</span>');

  const flagsHtml = c.flags.map(([f, d]) =>
    `<div class="flag-row"><code class="flag">${f}</code><div class="flag-desc">${d}</div></div>`
  ).join('');

  const outputHtml = c.output.map(line => {
    let l = line
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/-&gt;/g, '→');
    l = l
      .replace(/\b(PASS|verified|forged|identical|true|OK|HEALTHY)\b/g, '<span class="pass">$1</span>')
      .replace(/\b(FAIL|ERROR|WARN|ISSUES FOUND|false|corrupt)\b/g, '<span class="fail">$1</span>')
      .replace(/\b(INFO|saved|backup|patched)\b/g, '<span class="info">$1</span>')
      .replace(/(\b\d+\.\d+\b|\b\d{2,}\b|\b1e-?\d+\b)/g, '<span class="num">$1</span>')
      .replace(/(\[[a-z]+\])/g, '<span class="dim">$1</span>')
      .replace(/(\([^)]+\))/g, '<span class="dim">$1</span>')
      .replace(/(─+)/g, '<span class="dim">$1</span>')
      .replace(/(\$ )/g, '<span class="prompt-inline">$1</span>');
    return l;
  }).join('\n');

  return `<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
*{margin:0;padding:0;box-sizing:border-box}
body{width:1200px;height:630px;background:#0d1117;font-family:'Segoe UI',system-ui,sans-serif;color:#e6edf3;display:flex;flex-direction:column;padding:28px 36px}
.header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px}
.cat-badge{font-size:10px;font-weight:600;letter-spacing:.09em;color:${c.accent};text-transform:uppercase;border:1px solid ${c.accent}55;border-radius:4px;padding:3px 9px;display:inline-block;margin-bottom:8px}
.cmd-name{font-size:42px;font-weight:700;color:#e6edf3;font-family:'Cascadia Code','Fira Code','Consolas',monospace;margin-bottom:4px;line-height:1}
.cmd-name::before{content:'cjcl ';color:#8b949e;font-weight:500}
.subtitle{font-size:14px;color:#8b949e;line-height:1.4;max-width:760px}
.lang-pill{font-size:10px;color:#8b949e;background:#161b22;border:1px solid #30363d;border-radius:4px;padding:4px 9px;white-space:nowrap}

.body-grid{display:grid;grid-template-columns:1fr 360px;gap:14px;flex:1;min-height:0}

.terminal{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden;display:flex;flex-direction:column}
.term-bar{background:#1c2128;border-bottom:1px solid #30363d;padding:7px 14px;display:flex;align-items:center;gap:7px;flex-shrink:0}
.dot{width:10px;height:10px;border-radius:50%}
.term-title{font-size:10px;color:#8b949e;margin-left:5px;font-family:monospace}
.prompt{padding:10px 16px 4px 16px;font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:12.5px;line-height:1.4;flex-shrink:0}
.prompt::before{content:'$ ';color:#3fb950;font-weight:600}
.output{padding:2px 16px 12px 16px;font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:11.5px;line-height:1.5;color:#c9d1d9;white-space:pre;flex:1;overflow:hidden}
.output .dim{color:#6e7681}
.output .pass{color:#3fb950;font-weight:600}
.output .fail{color:#f85149;font-weight:600}
.output .info{color:#58a6ff;font-weight:600}
.output .num{color:#79c0ff}
.output .prompt-inline{color:#3fb950;font-weight:600}

.side{display:flex;flex-direction:column;gap:10px;min-height:0}
.panel{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:11px 13px}
.panel-title{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:${c.accent};margin-bottom:6px;display:flex;align-items:center;gap:6px}
.panel-title::before{content:'';width:3px;height:11px;background:${c.accent};border-radius:2px}
.what-text{font-size:11.5px;color:#c9d1d9;line-height:1.5}
.flags-panel{flex:1;overflow:hidden}
.flag-row{display:flex;align-items:flex-start;gap:8px;padding:4px 0;border-bottom:1px solid #21262d}
.flag-row:last-child{border-bottom:none}
.flag{font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:10.5px;color:#79c0ff;background:#0d1117;border:1px solid #30363d;padding:1px 6px;border-radius:3px;white-space:nowrap;flex-shrink:0;min-width:120px;text-align:left}
.flag-desc{font-size:10.5px;color:#8b949e;line-height:1.35;flex:1}

.note-pill{margin-top:10px;font-size:11.5px;color:${c.accent};background:#161b22;border:1px solid ${c.accent}44;border-radius:20px;padding:6px 14px;display:inline-flex;align-items:center;gap:7px;align-self:flex-start}
.note-pill::before{content:'→';color:${c.accent};font-weight:bold}
</style></head><body>
  <div class="header">
    <div>
      <div class="cat-badge">${c.cat}</div>
      <div class="cmd-name">${name}</div>
      <p class="subtitle">${c.desc}</p>
    </div>
    <div class="lang-pill">CJC-Lang · cjcl · v0.1.6</div>
  </div>

  <div class="body-grid">
    <div class="terminal">
      <div class="term-bar">
        <div class="dot" style="background:#ff5f57"></div>
        <div class="dot" style="background:#febc2e"></div>
        <div class="dot" style="background:#28c840"></div>
        <span class="term-title">~/project · cjcl ${name}</span>
      </div>
      <div class="prompt">${cmdHtml}</div>
      <div class="output">${outputHtml}</div>
    </div>

    <div class="side">
      <div class="panel">
        <div class="panel-title">What it does</div>
        <div class="what-text">${c.what}</div>
      </div>
      <div class="panel flags-panel">
        <div class="panel-title">Key flags</div>
        ${flagsHtml}
      </div>
    </div>
  </div>

  <div class="note-pill">${c.note}</div>
</body></html>`;
}

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 630, deviceScaleFactor: 2 });

  for (const [name, c] of Object.entries(COMMANDS)) {
    const html = buildHtml(name, c);
    await page.setContent(html, { waitUntil: 'load' });
    const out = path.join(__dirname, `cjcl_${name}.jpg`);
    await page.screenshot({ path: out, type: 'jpeg', quality: 95, clip: { x: 0, y: 0, width: 1200, height: 630 } });
    console.log('wrote', out);
  }

  await browser.close();
})();
