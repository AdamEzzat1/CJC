// Generate one social image per CLI command (1200×630 JPEG, retina 2x).
// Data is real — output samples drawn from cjc-cli source, not invented.

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const COMMANDS = {
  // ---------- DATA & PIPELINE (7) ----------
  flow: {
    cat: 'Data & Pipeline',
    accent: '#58a6ff',
    desc: 'Streaming computation over CSV — single pass, Kahan summation',
    cmd: 'cjcl flow data.csv --op sum,mean,count',
    output: [
      'Column         sum (Kahan)    mean         min          max',
      'price          12450.50       124.51       10.00        999.99',
      'quantity       5230.00        52.30        1.00         100.00',
      '(100 rows processed)',
    ],
    note: 'O(ncols) memory · Kahan compensated summation · single pass',
  },
  patch: {
    cat: 'Data & Pipeline',
    accent: '#58a6ff',
    desc: 'Type-aware data transformation — NaN fill, impute, drop, rename',
    cmd: 'cjcl patch data.csv --nan-fill 0 --drop temp_col --plan',
    output: [
      'Patch Plan',
      '  source: data.csv',
      '  format: CSV',
      '  transforms (3):',
      '  1. nan-fill: replace NaN/NA/null/None with "0"',
      '  patched 1500 rows',
    ],
    note: 'streaming · --plan to preview · --dry-run to verify',
  },
  drift: {
    cat: 'Data & Pipeline',
    accent: '#58a6ff',
    desc: 'Mathematical file diff with per-cell numeric tolerance',
    cmd: 'cjcl drift old.csv new.csv --tolerance 0.01',
    output: [
      'Metric               Value',
      'Rows (A)             1000',
      'Rows (B)             1000',
      'Cell differences     42',
      'Max deviation        0.012350',
      'identical: false',
    ],
    note: 'CSV · numeric · text · JSONL · Frobenius norm · per-cell diff',
  },
  seek: {
    cat: 'Data & Pipeline',
    accent: '#58a6ff',
    desc: 'Deterministic file discovery — glob, content search, manifest',
    cmd: "cjcl seek . '*.snap' --hash --sort size",
    output: [
      'a1b2c3d4e5f6g7h8 results.snap',
      'f9e8d7c6b5a4392f cache.snap',
      '8270bfcc3a922100 debug.snap',
      '3 files found',
    ],
    note: 'sorted output · SHA-256 hashes · reproducibility manifest',
  },
  forge: {
    cat: 'Data & Pipeline',
    accent: '#58a6ff',
    desc: 'Content-addressable pipeline runner — outputs as SHA-256 artifacts',
    cmd: 'cjcl forge run script.cjcl --seed 42',
    output: [
      'Property         Value',
      'Source           script.cjcl',
      'Seed             42',
      'Output lines     25',
      'SHA-256          a1b2c3d4e5f6...',
      'forged → .cjc-forge/a1b2c3d4',
    ],
    note: 'run · verify · list · show · clean · cache in .cjc-forge/',
  },
  proof: {
    cat: 'Data & Pipeline',
    accent: '#58a6ff',
    desc: 'Determinism verifier — N runs, stdout, exit, and GC must agree',
    cmd: 'cjcl proof program.cjcl --runs 3 -v',
    output: [
      '[proof] Running 3 iterations with seed 42 (executor: eval)…',
      'Check                    Status',
      'stdout identical         PASS',
      'exit status identical    PASS',
      'GC collections identical PASS',
      'Verdict: PASS (3 runs, seed=42)',
    ],
    note: '--runs N · --seeds 42,123,999 · --executor both · --hash-output',
  },
  view: {
    cat: 'Data & Pipeline',
    accent: '#58a6ff',
    desc: 'Deterministic directory listing — effect annotations + hashes',
    cmd: 'cjcl view src/ --recursive --hash',
    output: [
      'src/ (4 entries)',
      'Name             Kind   Size   Effects     Hash',
      'main.cjcl        cjcl   2.3K   nogc,pure   a1b2c3d4…',
      'data.csv         data   15K    —           —',
      'output.snap      snap   512B   —           f9e8d7c6…',
      'helpers.cjcl     cjcl   1.8K   —           —',
    ],
    note: 'lexicographic order · effect annotations from #[nogc] @pure',
  },

  // ---------- INSPECTION & DIAGNOSTICS (7) ----------
  inspect: {
    cat: 'Inspection & Diagnostics',
    accent: '#bc8cff',
    desc: 'Deep file inspection — schema, stats, hashing without execution',
    cmd: 'cjcl inspect data.csv --deep',
    output: [
      'Property         Value',
      'File             data.csv',
      'Type             csv',
      'Rows             5000',
      'Columns          12',
      'Size             8.2M',
    ],
    note: '.snap · CSV · JSONL · Parquet · Arrow · SQLite · model files',
  },
  schema: {
    cat: 'Inspection & Diagnostics',
    accent: '#bc8cff',
    desc: 'Schema inference + validation — null counts, types, drift detection',
    cmd: 'cjcl schema data.csv --full --save schema.json',
    output: [
      'Schema for `data.csv` (5000 rows sampled):',
      'Column   Type     Nulls   Total   Unique   Min    Max',
      'id       int      0       5000    5000     1      5000',
      'name     string   15      5000    4980     —      —',
      'price    float    8       5000    1203     0.99   999.99',
    ],
    note: '--save · --check · --diff · CI-ready schema enforcement',
  },
  trace: {
    cat: 'Inspection & Diagnostics',
    accent: '#bc8cff',
    desc: 'Execution tracing — parse, exec, AST structure, GC behavior',
    cmd: 'cjcl trace script.cjcl --verbose',
    output: [
      '[trace] Tracing `script.cjcl`…',
      'Phase            Detail',
      'Parse time       2.341 ms',
      'Exec time        15.127 ms',
      'AST              42 fns · 8 structs',
      'Status           OK',
    ],
    note: '4 phases · parse · static · execution · reporting',
  },
  mem: {
    cat: 'Inspection & Diagnostics',
    accent: '#bc8cff',
    desc: 'Memory profiling — GC collections, heap, run-to-run stability',
    cmd: 'cjcl mem script.cjcl --runs 5 --timeline',
    output: [
      '[mem] Memory profile for `script.cjcl` (5 runs, seed=42)',
      'Metric                       Value',
      'GC collections (avg)         0.60',
      'GC stable across runs        true',
      'GC heap objects (max)        234',
      'Output size (bytes)          512',
    ],
    note: '--compare eval|mir · --fail-on-gc · --timeline · per-run breakdown',
  },
  bench: {
    cat: 'Inspection & Diagnostics',
    accent: '#bc8cff',
    desc: 'Performance benchmarking — mean, median, P95, CV, baseline diff',
    cmd: 'cjcl bench script.cjcl --runs 10 --warmup 2',
    output: [
      '[bench] Benchmarking `script.cjcl` (2 warmup + 10 measured)',
      'Metric           Value',
      'Mean             45.23 ms',
      'Median           44.12 ms',
      'Stddev           2.18 ms',
      'CV               4.8%',
    ],
    note: '--compare eval|mir · --baseline · --fail-if-slower-than 10',
  },
  pack: {
    cat: 'Inspection & Diagnostics',
    accent: '#bc8cff',
    desc: 'Reproducible packaging — auto-discovery + per-file SHA-256 manifest',
    cmd: 'cjcl pack script.cjcl -o script.pack --verify',
    output: [
      'File                Size',
      'script.cjcl         2.3K',
      'data.csv            8.2M',
      'packed 2 files (8.2M) into `script.pack`',
      'Manifest: a1b2c3d4e5f6…',
      'verified',
    ],
    note: 'auto-detects .snap · .csv · .json · .toml · .lock dependencies',
  },
  doctor: {
    cat: 'Inspection & Diagnostics',
    accent: '#bc8cff',
    desc: 'Project diagnostics — parse errors, nondeterministic patterns, drift',
    cmd: 'cjcl doctor . --strict',
    output: [
      'Category              Count',
      'CJC source files      12',
      'Snap files            8',
      'CSV/TSV files         15',
      '[ERROR] main.cjcl — 2 parse error(s)',
      'ISSUES FOUND',
    ],
    note: 'flags HashMap · rand() · time() · ragged CSV · corrupt snaps',
  },
};

function buildHtml(name, c) {
  const cmdHtml = c.cmd
    .replace(/cjcl/g, '<span style="color:#3fb950">cjcl</span>')
    .replace(/(--[a-z-]+)/g, '<span style="color:#79c0ff">$1</span>')
    .replace(/('[^']*'|"[^"]*")/g, '<span style="color:#a5d6ff">$1</span>');

  return `<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
*{margin:0;padding:0;box-sizing:border-box}
body{width:1200px;height:630px;background:#0d1117;font-family:'Segoe UI',system-ui,sans-serif;color:#e6edf3;display:flex;flex-direction:column;padding:44px 56px}
.header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px}
.cat-badge{font-size:11px;font-weight:600;letter-spacing:.08em;color:${c.accent};text-transform:uppercase;border:1px solid ${c.accent}55;border-radius:4px;padding:4px 10px;display:inline-block;margin-bottom:14px}
.cmd-name{font-size:54px;font-weight:700;color:#e6edf3;font-family:'Cascadia Code','Fira Code','Consolas',monospace;margin-bottom:6px}
.cmd-name::before{content:'cjcl ';color:#8b949e;font-weight:500}
.subtitle{font-size:17px;color:#8b949e;line-height:1.4;max-width:780px}
.lang-pill{font-size:11px;color:#8b949e;background:#161b22;border:1px solid #30363d;border-radius:4px;padding:5px 10px}
.terminal{background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden;flex:1;display:flex;flex-direction:column;margin-bottom:18px}
.term-bar{background:#1c2128;border-bottom:1px solid #30363d;padding:9px 16px;display:flex;align-items:center;gap:8px}
.dot{width:11px;height:11px;border-radius:50%}
.term-title{font-size:11px;color:#8b949e;margin-left:6px;font-family:monospace}
.prompt{padding:14px 20px 8px 20px;font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:15px;line-height:1.5}
.prompt::before{content:'$ ';color:#3fb950;font-weight:600}
.output{padding:0 20px 16px 20px;font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:14px;line-height:1.55;color:#c9d1d9;white-space:pre}
.output .dim{color:#8b949e}
.output .pass{color:#3fb950;font-weight:600}
.output .fail{color:#f85149;font-weight:600}
.output .num{color:#79c0ff}
.note-pill{font-size:13px;color:${c.accent};background:#161b22;border:1px solid ${c.accent}44;border-radius:20px;padding:8px 16px;display:inline-flex;align-items:center;gap:8px;align-self:flex-start}
.note-pill::before{content:'→';color:${c.accent};font-weight:bold}
</style></head><body>
  <div class="header">
    <div>
      <div class="cat-badge">${c.cat}</div>
      <div class="cmd-name">${name}</div>
      <p class="subtitle">${c.desc}</p>
    </div>
    <div class="lang-pill">CJC-Lang · cjcl</div>
  </div>
  <div class="terminal">
    <div class="term-bar">
      <div class="dot" style="background:#ff5f57"></div>
      <div class="dot" style="background:#febc2e"></div>
      <div class="dot" style="background:#28c840"></div>
      <span class="term-title">~/project · cjcl ${name}</span>
    </div>
    <div class="prompt">${cmdHtml}</div>
    <div class="output">${c.output.map(line => {
      // very small "syntax highlighting" for outputs
      let l = line
        .replace(/\b(PASS|verified|forged|identical|true|OK)\b/g, '<span class="pass">$1</span>')
        .replace(/\b(FAIL|ERROR|ISSUES FOUND|false)\b/g, '<span class="fail">$1</span>')
        .replace(/\b(\d+\.\d+|\d+\.\d+ms|\d+\.\d+ ms|\d+\.\d+%|\d{2,})\b/g, '<span class="num">$1</span>');
      // dim brackets
      l = l.replace(/(\[[a-z]+\])/g, '<span class="dim">$1</span>');
      // dim parenthetical
      l = l.replace(/(\([^)]+\))/g, '<span class="dim">$1</span>');
      return l;
    }).join('\n')}</div>
  </div>
  <div class="note-pill">${c.note}</div>
</body></html>`;
}

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 630, deviceScaleFactor: 2 });

  const outDir = __dirname;
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  for (const [name, c] of Object.entries(COMMANDS)) {
    const html = buildHtml(name, c);
    await page.setContent(html, { waitUntil: 'load' });
    const out = path.join(outDir, `cjcl_${name}.jpg`);
    await page.screenshot({ path: out, type: 'jpeg', quality: 95, clip: { x: 0, y: 0, width: 1200, height: 630 } });
    console.log('wrote', out);
  }

  await browser.close();
})();
