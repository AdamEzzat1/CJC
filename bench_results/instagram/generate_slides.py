"""
CJC v0.1.3 — Instagram Carousel Generator
Generates 1080x1080 slides for Instagram post
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Color Palette (dark tech theme) ──────────────────────────────────
BG = '#0d1117'
BG2 = '#161b22'
CARD = '#1c2333'
ACCENT = '#58a6ff'
ACCENT2 = '#3fb950'
ACCENT3 = '#d29922'
ACCENT4 = '#f85149'
ACCENT5 = '#bc8cff'
TEXT = '#e6edf3'
TEXT2 = '#8b949e'
WHITE = '#ffffff'
GRID = '#21262d'

DPI = 200
SIZE = (5.4, 5.4)  # 1080x1080 at 200 DPI

def setup_slide(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

def add_logo(ax):
    ax.text(50, 96, 'CJC', fontsize=14, color=ACCENT, fontweight='bold',
            ha='center', va='top', fontfamily='monospace')
    ax.text(50, 92.5, 'Deterministic Scientific Computing', fontsize=6,
            color=TEXT2, ha='center', va='top')

def add_footer(ax, page, total):
    ax.plot([10, 90], [3.5, 3.5], color=GRID, linewidth=0.5)
    ax.text(50, 1.5, f'v0.1.3  •  {page}/{total}  •  cargo install cjc',
            fontsize=5.5, color=TEXT2, ha='center', fontfamily='monospace')

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 1: Hero / Title Card
# ═══════════════════════════════════════════════════════════════════════
def slide_1():
    fig, ax = plt.subplots(1, 1, figsize=SIZE, dpi=DPI)
    setup_slide(fig, ax)

    # Logo area
    ax.text(50, 82, 'CJC', fontsize=42, color=WHITE, fontweight='bold',
            ha='center', va='center', fontfamily='monospace')

    # Tagline
    ax.text(50, 73, 'A Deterministic Programming Language', fontsize=11,
            color=ACCENT, ha='center', va='center')
    ax.text(50, 68, 'for Scientific Computing & ML', fontsize=11,
            color=ACCENT, ha='center', va='center')

    # Divider
    ax.plot([25, 75], [62, 62], color=ACCENT, linewidth=1.5, alpha=0.6)

    # Stats grid
    stats = [
        ('8.7 MB', 'Binary Size'),
        ('30+', 'CLI Commands'),
        ('0', 'External Deps'),
        ('100%', 'Deterministic'),
    ]
    for i, (val, label) in enumerate(stats):
        x = 15 + (i % 2) * 38
        y = 48 - (i // 2) * 18
        # Card background
        rect = FancyBboxPatch((x-12, y-7), 32, 14, boxstyle="round,pad=1",
                              facecolor=CARD, edgecolor=ACCENT, linewidth=0.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+4, y+2, val, fontsize=16, color=WHITE, fontweight='bold',
                ha='center', va='center', fontfamily='monospace')
        ax.text(x+4, y-3.5, label, fontsize=6, color=TEXT2,
                ha='center', va='center')

    # Bottom
    ax.text(50, 11, 'cargo install cjc', fontsize=10, color=ACCENT2,
            ha='center', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=BG2, edgecolor=ACCENT2, linewidth=0.8))

    ax.text(50, 5, 'Built in Rust  •  Zero Dependencies  •  MIT License', fontsize=5.5,
            color=TEXT2, ha='center')

    fig.savefig('C:/Users/adame/CJC/bench_results/instagram/slide_01_hero.png',
                bbox_inches='tight', pad_inches=0.1, facecolor=BG)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 2: 30 CLI Commands Overview
# ═══════════════════════════════════════════════════════════════════════
def slide_2():
    fig, ax = plt.subplots(1, 1, figsize=SIZE, dpi=DPI)
    setup_slide(fig, ax)
    add_logo(ax)

    ax.text(50, 87, '30 CLI Commands — Zero Dependencies', fontsize=9,
            color=WHITE, fontweight='bold', ha='center')

    categories = [
        ('CORE', ACCENT, ['lex', 'parse', 'check', 'run', 'eval', 'repl']),
        ('DATA PIPELINE', ACCENT2, ['flow', 'schema', 'inspect', 'patch', 'drift', 'seek', 'forge']),
        ('DIAGNOSTICS', ACCENT3, ['trace', 'mem', 'bench', 'doctor', 'pack', 'view']),
        ('COMPILER', ACCENT5, ['emit', 'explain', 'gc', 'nogc', 'audit', 'precision']),
        ('VERIFICATION', ACCENT4, ['proof', 'parity', 'lock', 'test', 'ci']),
    ]

    y = 81
    for cat_name, color, cmds in categories:
        ax.text(8, y, cat_name, fontsize=6, color=color, fontweight='bold',
                fontfamily='monospace')
        y -= 2

        # Command pills
        x = 8
        for cmd in cmds:
            w = len(cmd) * 1.8 + 4
            rect = FancyBboxPatch((x, y-2.2), w, 4,
                                  boxstyle="round,pad=0.3",
                                  facecolor=BG2, edgecolor=color, linewidth=0.4)
            ax.add_patch(rect)
            ax.text(x + w/2, y, cmd, fontsize=5, color=TEXT,
                    ha='center', va='center', fontfamily='monospace')
            x += w + 2
            if x > 85:
                x = 8
                y -= 5

        y -= 6

    ax.text(50, 7, 'Every command: --plain | --json | --color output modes',
            fontsize=5.5, color=TEXT2, ha='center', fontfamily='monospace')

    add_footer(ax, 2, 8)
    fig.savefig('C:/Users/adame/CJC/bench_results/instagram/slide_02_commands.png',
                bbox_inches='tight', pad_inches=0.1, facecolor=BG)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 3: Data Pipeline Performance
# ═══════════════════════════════════════════════════════════════════════
def slide_3():
    fig, ax = plt.subplots(1, 1, figsize=SIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'CJC', fontsize=12, color=ACCENT, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes, fontfamily='monospace')
    ax.text(0.5, 0.92, 'Data Pipeline — 1M Rows (75 MB CSV)', fontsize=9,
            color=WHITE, fontweight='bold', ha='center', transform=ax.transAxes)

    commands = ['flow', 'flow\n--verify', 'schema', 'schema\n--full', 'inspect', 'inspect\n--deep', 'inspect\n--hash', 'patch\n--dry-run']
    times_ms = [956, 1483, 1659, 1653, 1199, 7267, 2008, 104]

    colors = [ACCENT if t < 2000 else ACCENT3 if t < 5000 else ACCENT4 for t in times_ms]

    bars = ax.barh(range(len(commands)), times_ms, color=colors, height=0.6, alpha=0.85,
                   edgecolor=[c for c in colors], linewidth=0.5)

    ax.set_yticks(range(len(commands)))
    ax.set_yticklabels(commands, fontsize=6, color=TEXT, fontfamily='monospace')
    ax.invert_yaxis()

    for i, (bar, t) in enumerate(zip(bars, times_ms)):
        if t >= 1000:
            label = f'{t/1000:.1f}s'
        else:
            label = f'{t}ms'
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                label, fontsize=6, color=TEXT, va='center', fontfamily='monospace')

    ax.set_xlabel('Time (ms)', fontsize=7, color=TEXT2, fontfamily='monospace')
    ax.tick_params(axis='x', colors=TEXT2, labelsize=5)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['left'].set_color(GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(times_ms) * 1.25)

    ax.text(0.5, 0.04, 'All results: 100% DETERMINISTIC  •  Kahan summation  •  Zero alloc overhead',
            fontsize=5, color=ACCENT2, ha='center', transform=ax.transAxes, fontfamily='monospace')

    fig.savefig('C:/Users/adame/CJC/bench_results/instagram/slide_03_data_pipeline.png',
                bbox_inches='tight', pad_inches=0.3, facecolor=BG)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 4: Eval vs MIR-opt Speed Comparison
# ═══════════════════════════════════════════════════════════════════════
def slide_4():
    fig, ax = plt.subplots(1, 1, figsize=SIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'CJC', fontsize=12, color=ACCENT, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes, fontfamily='monospace')
    ax.text(0.5, 0.92, 'Dual Executor — Eval vs MIR-Opt', fontsize=9,
            color=WHITE, fontweight='bold', ha='center', transform=ax.transAxes)

    labels = ['100K\n(arithmetic + fib + tensor)', '1M\n(arithmetic + fib + tensor + matmul)']
    eval_times = [6673, 8304]
    mir_times = [2257, 3732]

    x = np.arange(len(labels))
    width = 0.3

    bars1 = ax.bar(x - width/2, eval_times, width, label='Eval (tree-walk)',
                   color=ACCENT3, alpha=0.85, edgecolor=ACCENT3, linewidth=0.5)
    bars2 = ax.bar(x + width/2, mir_times, width, label='MIR-Opt (register machine)',
                   color=ACCENT2, alpha=0.85, edgecolor=ACCENT2, linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                f'{bar.get_height()/1000:.1f}s', ha='center', fontsize=7,
                color=ACCENT3, fontfamily='monospace', fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                f'{bar.get_height()/1000:.1f}s', ha='center', fontsize=7,
                color=ACCENT2, fontfamily='monospace', fontweight='bold')

    # Speedup annotations
    for i in range(len(labels)):
        speedup = eval_times[i] / mir_times[i]
        ax.annotate(f'{speedup:.1f}x faster',
                    xy=(x[i] + width/2, mir_times[i]),
                    xytext=(x[i] + 0.55, mir_times[i] + 1500),
                    fontsize=6, color=ACCENT2, fontweight='bold',
                    fontfamily='monospace',
                    arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5, color=TEXT, fontfamily='monospace')
    ax.set_ylabel('Time (ms)', fontsize=7, color=TEXT2, fontfamily='monospace')
    ax.tick_params(axis='y', colors=TEXT2, labelsize=5)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['left'].set_color(GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6, loc='upper left', facecolor=CARD, edgecolor=GRID,
              labelcolor=TEXT, framealpha=0.9)

    ax.text(0.5, 0.04, 'Both executors produce BIT-IDENTICAL output across all seeds',
            fontsize=5.5, color=ACCENT, ha='center', transform=ax.transAxes, fontfamily='monospace')

    fig.savefig('C:/Users/adame/CJC/bench_results/instagram/slide_04_eval_vs_mir.png',
                bbox_inches='tight', pad_inches=0.3, facecolor=BG)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 5: 100% Determinism Proof
# ═══════════════════════════════════════════════════════════════════════
def slide_5():
    fig, ax = plt.subplots(1, 1, figsize=SIZE, dpi=DPI)
    setup_slide(fig, ax)
    add_logo(ax)

    ax.text(50, 87, '100% Deterministic — Every Seed, Every Run', fontsize=8.5,
            color=WHITE, fontweight='bold', ha='center')

    # SHA-256 verification grid
    seeds = [1, 42, 100, 999, 12345]
    hashes_100k = ['08d4ccd0', '93e279c7', '4a52508e', '3a8f0eef', 'c01eadef']
    hashes_1m = ['20e7987f', 'aeb48bad', 'e41c6b48', '28c442cf', '0e5fb74a']

    # Table header
    y = 79
    ax.text(12, y, 'SEED', fontsize=5.5, color=TEXT2, fontweight='bold', fontfamily='monospace')
    ax.text(30, y, '100K HASH', fontsize=5.5, color=TEXT2, fontweight='bold', fontfamily='monospace')
    ax.text(57, y, '1M HASH', fontsize=5.5, color=TEXT2, fontweight='bold', fontfamily='monospace')
    ax.text(82, y, 'PARITY', fontsize=5.5, color=TEXT2, fontweight='bold', fontfamily='monospace')

    ax.plot([8, 92], [y-1.5, y-1.5], color=GRID, linewidth=0.5)

    for i, (seed, h100, h1m) in enumerate(zip(seeds, hashes_100k, hashes_1m)):
        y = 74 - i * 7
        # Row bg
        if i % 2 == 0:
            rect = patches.Rectangle((7, y-2.5), 86, 5.5, facecolor=BG2, alpha=0.3)
            ax.add_patch(rect)

        ax.text(15, y, str(seed), fontsize=6.5, color=ACCENT, fontweight='bold',
                ha='center', fontfamily='monospace')
        ax.text(30, y+1, f'Run 1: {h100}...', fontsize=4.8, color=TEXT,
                fontfamily='monospace')
        ax.text(30, y-2, f'Run 2: {h100}...', fontsize=4.8, color=TEXT,
                fontfamily='monospace')
        ax.text(57, y+1, f'Eval:    {h1m}...', fontsize=4.8, color=TEXT,
                fontfamily='monospace')
        ax.text(57, y-2, f'MIR-Opt: {h1m}...', fontsize=4.8, color=TEXT,
                fontfamily='monospace')

        # Checkmark
        ax.text(85, y, '✓', fontsize=12, color=ACCENT2, ha='center', va='center',
                fontweight='bold')

    # Summary box
    rect = FancyBboxPatch((15, 7), 70, 10, boxstyle="round,pad=1",
                          facecolor=CARD, edgecolor=ACCENT2, linewidth=1)
    ax.add_patch(rect)
    ax.text(50, 13.5, '20/20 seeds MATCH  •  10/10 EVAL=MIR  •  0 divergences',
            fontsize=6, color=ACCENT2, ha='center', fontweight='bold', fontfamily='monospace')
    ax.text(50, 9.5, 'Same seed = bit-identical output, guaranteed.',
            fontsize=5.5, color=TEXT2, ha='center', fontfamily='monospace')

    add_footer(ax, 5, 8)
    fig.savefig('C:/Users/adame/CJC/bench_results/instagram/slide_05_determinism.png',
                bbox_inches='tight', pad_inches=0.1, facecolor=BG)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 6: Scaling — 100K vs 1M
# ═══════════════════════════════════════════════════════════════════════
def slide_6():
    fig, ax = plt.subplots(1, 1, figsize=SIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'CJC', fontsize=12, color=ACCENT, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes, fontfamily='monospace')
    ax.text(0.5, 0.92, 'Linear Scaling — 100K to 1M Rows', fontsize=9,
            color=WHITE, fontweight='bold', ha='center', transform=ax.transAxes)

    commands = ['flow', 'schema', 'inspect', 'inspect\n--deep', 'inspect\n--hash', 'schema\n--sample 1000']
    times_100k = [128, 203, 176, 803, 220, 88]
    times_1m = [956, 1659, 1199, 7267, 2008, 205]
    ratios = [t1m/t100k for t100k, t1m in zip(times_100k, times_1m)]

    x = np.arange(len(commands))
    width = 0.3

    bars1 = ax.bar(x - width/2, times_100k, width, label='100K rows (7.4 MB)',
                   color=ACCENT, alpha=0.8)
    bars2 = ax.bar(x + width/2, times_1m, width, label='1M rows (75 MB)',
                   color=ACCENT5, alpha=0.8)

    # Scaling labels
    for i, ratio in enumerate(ratios):
        y_pos = max(times_100k[i], times_1m[i]) + 300
        color = ACCENT2 if ratio < 10 else ACCENT3
        ax.text(x[i], y_pos, f'{ratio:.1f}x', ha='center', fontsize=6,
                color=color, fontweight='bold', fontfamily='monospace')

    ax.set_xticks(x)
    ax.set_xticklabels(commands, fontsize=5, color=TEXT, fontfamily='monospace')
    ax.set_ylabel('Time (ms)', fontsize=7, color=TEXT2, fontfamily='monospace')
    ax.tick_params(axis='y', colors=TEXT2, labelsize=5)
    ax.tick_params(axis='x', colors=TEXT2)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['left'].set_color(GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6, loc='upper left', facecolor=CARD, edgecolor=GRID,
              labelcolor=TEXT, framealpha=0.9)

    ax.text(0.5, 0.04, '10x data → ~7-9x time  •  Near-linear scaling  •  O(n) streaming',
            fontsize=5.5, color=ACCENT2, ha='center', transform=ax.transAxes, fontfamily='monospace')

    fig.savefig('C:/Users/adame/CJC/bench_results/instagram/slide_06_scaling.png',
                bbox_inches='tight', pad_inches=0.3, facecolor=BG)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 7: Static Analysis Speed
# ═══════════════════════════════════════════════════════════════════════
def slide_7():
    fig, ax = plt.subplots(1, 1, figsize=SIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'CJC', fontsize=12, color=ACCENT, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes, fontfamily='monospace')
    ax.text(0.5, 0.92, 'Instant Static Analysis — Under 200ms', fontsize=9,
            color=WHITE, fontweight='bold', ha='center', transform=ax.transAxes)

    commands = ['lex', 'parse', 'emit\n--ast', 'emit\n--hir', 'emit\n--mir',
                'emit\n--mir --opt', 'explain', 'explain\n--verbose',
                'audit', 'audit\n--json', 'nogc', 'nogc\n--verbose']
    times = [87, 102, 75, 87, 103, 69, 76, 140, 75, 82, 75, 76]

    colors = [ACCENT2 if t < 100 else ACCENT for t in times]

    bars = ax.barh(range(len(commands)), times, color=colors, height=0.6, alpha=0.85)

    ax.set_yticks(range(len(commands)))
    ax.set_yticklabels(commands, fontsize=5, color=TEXT, fontfamily='monospace')
    ax.invert_yaxis()

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                f'{t}ms', fontsize=5.5, color=TEXT, va='center', fontfamily='monospace')

    ax.set_xlabel('Time (ms)', fontsize=7, color=TEXT2, fontfamily='monospace')
    ax.tick_params(axis='x', colors=TEXT2, labelsize=5)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['left'].set_color(GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 220)

    # Reference line at 100ms
    ax.axvline(x=100, color=ACCENT3, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.text(102, -0.5, '100ms', fontsize=4.5, color=ACCENT3, fontfamily='monospace')

    ax.text(0.5, 0.04, 'Full pipeline: Lex → Parse → AST → HIR → MIR → Optimize — all under 200ms',
            fontsize=5, color=ACCENT2, ha='center', transform=ax.transAxes, fontfamily='monospace')

    fig.savefig('C:/Users/adame/CJC/bench_results/instagram/slide_07_static_analysis.png',
                bbox_inches='tight', pad_inches=0.3, facecolor=BG)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════
# SLIDE 8: Architecture + CTA
# ═══════════════════════════════════════════════════════════════════════
def slide_8():
    fig, ax = plt.subplots(1, 1, figsize=SIZE, dpi=DPI)
    setup_slide(fig, ax)

    ax.text(50, 92, 'CJC', fontsize=28, color=WHITE, fontweight='bold',
            ha='center', fontfamily='monospace')
    ax.text(50, 86, 'Architecture', fontsize=10, color=ACCENT, ha='center')

    # Pipeline diagram
    stages = ['Lexer', 'Parser', 'AST', 'HIR', 'MIR', 'Optimize', 'Execute']
    colors_s = [ACCENT, ACCENT, ACCENT5, ACCENT5, ACCENT2, ACCENT2, ACCENT3]

    y = 74
    for i, (stage, col) in enumerate(zip(stages, colors_s)):
        x = 10 + i * 12
        rect = FancyBboxPatch((x-4, y-3), 10, 6, boxstyle="round,pad=0.3",
                              facecolor=CARD, edgecolor=col, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x+1, y, stage, fontsize=4.5, color=col, ha='center',
                va='center', fontfamily='monospace', fontweight='bold')
        if i < len(stages) - 1:
            ax.annotate('', xy=(x+7, y), xytext=(x+9, y),
                        arrowprops=dict(arrowstyle='->', color=GRID, lw=0.8))

    # Feature highlights
    features = [
        ('21 Rust crates', 'Modular workspace architecture'),
        ('Dual executor', 'AST eval + MIR register machine'),
        ('Tensor engine', 'matmul, conv2d, pooling, autodiff'),
        ('Data pipeline', 'DataFrame, filter, group_by, join'),
        ('Deterministic RNG', 'SplitMix64 + Kahan summation'),
        ('Zero deps', 'No external crates — pure Rust'),
    ]

    y = 62
    for i, (title, desc) in enumerate(features):
        col_x = 12 if i % 2 == 0 else 55
        if i > 0 and i % 2 == 0:
            y -= 10

        rect = FancyBboxPatch((col_x-3, y-3.5), 40, 7.5,
                              boxstyle="round,pad=0.5",
                              facecolor=BG2, edgecolor=GRID, linewidth=0.3)
        ax.add_patch(rect)
        ax.text(col_x, y+1, title, fontsize=5.5, color=ACCENT,
                fontweight='bold', fontfamily='monospace')
        ax.text(col_x, y-2, desc, fontsize=4.5, color=TEXT2, fontfamily='monospace')

    # CTA
    rect = FancyBboxPatch((18, 11), 64, 12, boxstyle="round,pad=1",
                          facecolor=CARD, edgecolor=ACCENT, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(50, 19, 'cargo install cjc', fontsize=11, color=ACCENT2,
            ha='center', fontweight='bold', fontfamily='monospace')
    ax.text(50, 14, 'github.com/AdamEzzat1/CJC', fontsize=6.5, color=ACCENT,
            ha='center', fontfamily='monospace')

    ax.text(50, 5, 'MIT License  •  Rust  •  Cross-platform', fontsize=5.5,
            color=TEXT2, ha='center')

    add_footer(ax, 8, 8)
    fig.savefig('C:/Users/adame/CJC/bench_results/instagram/slide_08_architecture.png',
                bbox_inches='tight', pad_inches=0.1, facecolor=BG)
    plt.close()

# ── Generate all slides ──────────────────────────────────────────────
print("Generating Instagram carousel slides...")
slide_1()
print("  ✓ Slide 1: Hero")
slide_2()
print("  ✓ Slide 2: Commands")
slide_3()
print("  ✓ Slide 3: Data Pipeline")
slide_4()
print("  ✓ Slide 4: Eval vs MIR")
slide_5()
print("  ✓ Slide 5: Determinism")
slide_6()
print("  ✓ Slide 6: Scaling")
slide_7()
print("  ✓ Slide 7: Static Analysis")
slide_8()
print("  ✓ Slide 8: Architecture + CTA")
print("\nAll 8 slides saved to bench_results/instagram/")
