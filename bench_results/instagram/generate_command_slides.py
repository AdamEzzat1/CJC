#!/usr/bin/env python3
"""Generate Instagram-ready terminal mockup images for CJC CLI data commands."""

import struct
import zlib
import os
import math

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Image dimensions (Instagram square)
W, H = 1080, 1080

# Color palette (RGB)
BG = (13, 17, 23)           # Dark terminal background
TITLE_BG = (22, 27, 34)     # Slightly lighter header bar
ACCENT = (88, 166, 255)     # Blue accent
GREEN = (63, 185, 80)       # Green for success/values
YELLOW = (210, 153, 34)     # Yellow for warnings/filenames
RED = (248, 81, 73)         # Red for errors/mismatches
CYAN = (57, 211, 215)       # Cyan for types/info
WHITE = (230, 237, 243)     # Main text
DIM = (125, 133, 144)       # Dimmed text
ORANGE = (219, 150, 68)     # Orange for numbers
PURPLE = (188, 140, 255)    # Purple for keywords
MAGENTA = (219, 97, 162)    # Pink/magenta for special

# ─── Minimal PNG writer (no dependencies) ───

def make_png(pixels, width, height):
    """Create PNG from flat RGB pixel list."""
    def chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)

    header = b'\x89PNG\r\n\x1a\n'
    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter byte
        for x in range(width):
            idx = (y * width + x) * 3
            raw.append(pixels[idx])
            raw.append(pixels[idx + 1])
            raw.append(pixels[idx + 2])

    idat = chunk(b'IDAT', zlib.compress(bytes(raw), 9))
    iend = chunk(b'IEND', b'')
    return header + ihdr + idat + iend


# ─── Bitmap font (5x7 pixel font) ───

FONT = {
    'A': ["01110","10001","10001","11111","10001","10001","10001"],
    'B': ["11110","10001","10001","11110","10001","10001","11110"],
    'C': ["01110","10001","10000","10000","10000","10001","01110"],
    'D': ["11100","10010","10001","10001","10001","10010","11100"],
    'E': ["11111","10000","10000","11110","10000","10000","11111"],
    'F': ["11111","10000","10000","11110","10000","10000","10000"],
    'G': ["01110","10001","10000","10111","10001","10001","01110"],
    'H': ["10001","10001","10001","11111","10001","10001","10001"],
    'I': ["01110","00100","00100","00100","00100","00100","01110"],
    'J': ["00111","00010","00010","00010","00010","10010","01100"],
    'K': ["10001","10010","10100","11000","10100","10010","10001"],
    'L': ["10000","10000","10000","10000","10000","10000","11111"],
    'M': ["10001","11011","10101","10101","10001","10001","10001"],
    'N': ["10001","10001","11001","10101","10011","10001","10001"],
    'O': ["01110","10001","10001","10001","10001","10001","01110"],
    'P': ["11110","10001","10001","11110","10000","10000","10000"],
    'Q': ["01110","10001","10001","10001","10101","10010","01101"],
    'R': ["11110","10001","10001","11110","10100","10010","10001"],
    'S': ["01111","10000","10000","01110","00001","00001","11110"],
    'T': ["11111","00100","00100","00100","00100","00100","00100"],
    'U': ["10001","10001","10001","10001","10001","10001","01110"],
    'V': ["10001","10001","10001","10001","01010","01010","00100"],
    'W': ["10001","10001","10001","10101","10101","10101","01010"],
    'X': ["10001","10001","01010","00100","01010","10001","10001"],
    'Y': ["10001","10001","01010","00100","00100","00100","00100"],
    'Z': ["11111","00001","00010","00100","01000","10000","11111"],
    'a': ["00000","00000","01110","00001","01111","10001","01111"],
    'b': ["10000","10000","10110","11001","10001","10001","11110"],
    'c': ["00000","00000","01110","10000","10000","10001","01110"],
    'd': ["00001","00001","01101","10011","10001","10001","01111"],
    'e': ["00000","00000","01110","10001","11111","10000","01110"],
    'f': ["00110","01001","01000","11100","01000","01000","01000"],
    'g': ["00000","01111","10001","10001","01111","00001","01110"],
    'h': ["10000","10000","10110","11001","10001","10001","10001"],
    'i': ["00100","00000","01100","00100","00100","00100","01110"],
    'j': ["00010","00000","00110","00010","00010","10010","01100"],
    'k': ["10000","10000","10010","10100","11000","10100","10010"],
    'l': ["01100","00100","00100","00100","00100","00100","01110"],
    'm': ["00000","00000","11010","10101","10101","10001","10001"],
    'n': ["00000","00000","10110","11001","10001","10001","10001"],
    'o': ["00000","00000","01110","10001","10001","10001","01110"],
    'p': ["00000","00000","11110","10001","11110","10000","10000"],
    'q': ["00000","00000","01101","10011","01111","00001","00001"],
    'r': ["00000","00000","10110","11001","10000","10000","10000"],
    's': ["00000","00000","01110","10000","01110","00001","11110"],
    't': ["01000","01000","11100","01000","01000","01001","00110"],
    'u': ["00000","00000","10001","10001","10001","10011","01101"],
    'v': ["00000","00000","10001","10001","10001","01010","00100"],
    'w': ["00000","00000","10001","10001","10101","10101","01010"],
    'x': ["00000","00000","10001","01010","00100","01010","10001"],
    'y': ["00000","00000","10001","10001","01111","00001","01110"],
    'z': ["00000","00000","11111","00010","00100","01000","11111"],
    '0': ["01110","10001","10011","10101","11001","10001","01110"],
    '1': ["00100","01100","00100","00100","00100","00100","01110"],
    '2': ["01110","10001","00001","00010","00100","01000","11111"],
    '3': ["11111","00010","00100","00010","00001","10001","01110"],
    '4': ["00010","00110","01010","10010","11111","00010","00010"],
    '5': ["11111","10000","11110","00001","00001","10001","01110"],
    '6': ["00110","01000","10000","11110","10001","10001","01110"],
    '7': ["11111","00001","00010","00100","01000","01000","01000"],
    '8': ["01110","10001","10001","01110","10001","10001","01110"],
    '9': ["01110","10001","10001","01111","00001","00010","01100"],
    ' ': ["00000","00000","00000","00000","00000","00000","00000"],
    '.': ["00000","00000","00000","00000","00000","00000","00100"],
    ',': ["00000","00000","00000","00000","00000","00100","01000"],
    ':': ["00000","00000","00100","00000","00000","00100","00000"],
    ';': ["00000","00000","00100","00000","00000","00100","01000"],
    '-': ["00000","00000","00000","11111","00000","00000","00000"],
    '_': ["00000","00000","00000","00000","00000","00000","11111"],
    '+': ["00000","00100","00100","11111","00100","00100","00000"],
    '=': ["00000","00000","11111","00000","11111","00000","00000"],
    '!': ["00100","00100","00100","00100","00100","00000","00100"],
    '?': ["01110","10001","00001","00010","00100","00000","00100"],
    '/': ["00001","00010","00010","00100","01000","01000","10000"],
    '\\':["10000","01000","01000","00100","00010","00010","00001"],
    '(': ["00010","00100","01000","01000","01000","00100","00010"],
    ')': ["01000","00100","00010","00010","00010","00100","01000"],
    '[': ["01110","01000","01000","01000","01000","01000","01110"],
    ']': ["01110","00010","00010","00010","00010","00010","01110"],
    '{': ["00110","00100","00100","01000","00100","00100","00110"],
    '}': ["01100","00100","00100","00010","00100","00100","01100"],
    '<': ["00010","00100","01000","10000","01000","00100","00010"],
    '>': ["01000","00100","00010","00001","00010","00100","01000"],
    '#': ["01010","01010","11111","01010","11111","01010","01010"],
    '$': ["00100","01111","10100","01110","00101","11110","00100"],
    '%': ["11001","11001","00010","00100","01000","10011","10011"],
    '&': ["01100","10010","10100","01000","10101","10010","01101"],
    '*': ["00000","00100","10101","01110","10101","00100","00000"],
    '@': ["01110","10001","10111","10101","10110","10000","01110"],
    '~': ["00000","00000","01000","10101","00010","00000","00000"],
    '`': ["01000","00100","00010","00000","00000","00000","00000"],
    "'": ["00100","00100","01000","00000","00000","00000","00000"],
    '"': ["01010","01010","01010","00000","00000","00000","00000"],
    '|': ["00100","00100","00100","00100","00100","00100","00100"],
    '^': ["00100","01010","10001","00000","00000","00000","00000"],
}

class Canvas:
    def __init__(self, w=W, h=H):
        self.w = w
        self.h = h
        self.pixels = bytearray(w * h * 3)
        self.fill_rect(0, 0, w, h, BG)

    def set_pixel(self, x, y, color):
        if 0 <= x < self.w and 0 <= y < self.h:
            idx = (y * self.w + x) * 3
            self.pixels[idx] = color[0]
            self.pixels[idx + 1] = color[1]
            self.pixels[idx + 2] = color[2]

    def fill_rect(self, x, y, w, h, color):
        for py in range(max(0, y), min(self.h, y + h)):
            for px in range(max(0, x), min(self.w, x + w)):
                idx = (py * self.w + px) * 3
                self.pixels[idx] = color[0]
                self.pixels[idx + 1] = color[1]
                self.pixels[idx + 2] = color[2]

    def fill_rounded_rect(self, x, y, w, h, r, color):
        """Fill rectangle with rounded corners."""
        for py in range(max(0, y), min(self.h, y + h)):
            for px in range(max(0, x), min(self.w, x + w)):
                # Check corners
                draw = True
                # Top-left
                if px < x + r and py < y + r:
                    if (px - (x + r))**2 + (py - (y + r))**2 > r*r:
                        draw = False
                # Top-right
                if px > x + w - r and py < y + r:
                    if (px - (x + w - r))**2 + (py - (y + r))**2 > r*r:
                        draw = False
                # Bottom-left
                if px < x + r and py > y + h - r:
                    if (px - (x + r))**2 + (py - (y + h - r))**2 > r*r:
                        draw = False
                # Bottom-right
                if px > x + w - r and py > y + h - r:
                    if (px - (x + w - r))**2 + (py - (y + h - r))**2 > r*r:
                        draw = False
                if draw:
                    idx = (py * self.w + px) * 3
                    self.pixels[idx] = color[0]
                    self.pixels[idx + 1] = color[1]
                    self.pixels[idx + 2] = color[2]

    def draw_char(self, x, y, ch, color, scale=2):
        glyph = FONT.get(ch)
        if glyph is None:
            return 6 * scale  # space width for unknown chars
        for row_idx, row in enumerate(glyph):
            for col_idx, bit in enumerate(row):
                if bit == '1':
                    for sy in range(scale):
                        for sx in range(scale):
                            self.set_pixel(x + col_idx * scale + sx, y + row_idx * scale + sy, color)
        return 6 * scale  # 5px char + 1px gap, scaled

    def draw_text(self, x, y, text, color, scale=2):
        cx = x
        for ch in text:
            cx += self.draw_char(cx, y, ch, color, scale)
        return cx

    def draw_text_centered(self, y, text, color, scale=2):
        text_w = len(text) * 6 * scale
        x = (self.w - text_w) // 2
        self.draw_text(x, y, text, color, scale)

    def draw_hline(self, x, y, w, color, thickness=1):
        for t in range(thickness):
            for px in range(x, x + w):
                self.set_pixel(px, y + t, color)

    def save(self, path):
        data = make_png(bytes(self.pixels), self.w, self.h)
        with open(path, 'wb') as f:
            f.write(data)
        print(f"  Saved: {path}")


def draw_terminal_frame(canvas, x, y, w, h, title="Terminal"):
    """Draw a terminal window frame with title bar."""
    # Title bar
    canvas.fill_rounded_rect(x, y, w, 36, 8, (40, 44, 52))
    # Window buttons
    canvas.fill_rect(x + 14, y + 12, 12, 12, (255, 95, 86))   # red
    canvas.fill_rect(x + 32, y + 12, 12, 12, (255, 189, 46))  # yellow
    canvas.fill_rect(x + 50, y + 12, 12, 12, (39, 201, 63))   # green
    # Title text
    title_w = len(title) * 12
    canvas.draw_text(x + (w - title_w) // 2, y + 10, title, DIM, 2)
    # Terminal body
    canvas.fill_rounded_rect(x, y + 36, w, h - 36, 8, (22, 27, 34))
    return y + 48  # content start y


def draw_table(canvas, x, y, headers, rows, col_widths, header_color=ACCENT, row_colors=None):
    """Draw a formatted table."""
    scale = 2
    row_h = 22
    char_w = 6 * scale

    # Header
    cx = x
    for i, hdr in enumerate(headers):
        canvas.draw_text(cx, y, hdr, header_color, scale)
        cx += col_widths[i] * char_w
    y += row_h

    # Separator
    total_w = sum(col_widths) * char_w
    canvas.draw_hline(x, y, total_w, DIM, 1)
    y += 6

    # Rows
    for ri, row in enumerate(rows):
        cx = x
        for ci, cell in enumerate(row):
            if row_colors and ci < len(row_colors):
                color = row_colors[ci]
            else:
                color = WHITE
            canvas.draw_text(cx, y, cell, color, scale)
            cx += col_widths[ci] * char_w
        y += row_h
    return y


def generate_flow():
    """Generate flow command slide."""
    c = Canvas()

    # Background gradient accent
    for py in range(40):
        alpha = 1.0 - py / 40.0
        r = int(BG[0] + (ACCENT[0] - BG[0]) * alpha * 0.3)
        g = int(BG[1] + (ACCENT[1] - BG[1]) * alpha * 0.3)
        b = int(BG[2] + (ACCENT[2] - BG[2]) * alpha * 0.3)
        c.fill_rect(0, py, W, 1, (r, g, b))

    # Title
    c.draw_text(40, 16, "cjcl flow", ACCENT, 3)
    c.draw_text(290, 22, "Streaming Stats Engine", DIM, 2)

    # Command box
    ty = draw_terminal_frame(c, 30, 60, W - 60, 130, "Terminal")
    c.draw_text(50, ty, "$ cjcl flow sales_data.csv --op sum,mean,var,std", GREEN, 2)
    c.draw_text(50, ty + 24, "  --top 5 --verify --precision 4", GREEN, 2)
    c.draw_text(50, ty + 56, "(1,000,000 rows processed, verified deterministic)", DIM, 2)

    # Output table
    oy = draw_terminal_frame(c, 30, 210, W - 60, 380, "Output")

    headers = ["Column", "sum(Kahan)", "mean", "var", "std"]
    rows = [
        ["revenue",   "48523190.12", "48.52",  "892.41",   "29.87"],
        ["quantity",  "2847561.00",  "2.85",   "4.12",     "2.03"],
        ["discount",  "125890.45",   "0.13",   "0.02",     "0.13"],
        ["tax_rate",  "89012.30",    "0.089",  "0.0008",   "0.028"],
        ["shipping",  "3421098.67",  "3.42",   "2.15",     "1.47"],
    ]
    col_colors = [YELLOW, ORANGE, ORANGE, ORANGE, ORANGE]

    draw_table(c, 50, oy, headers, rows, [12, 14, 8, 10, 8], CYAN, col_colors)

    # Verify badge
    vy = 530
    c.fill_rounded_rect(50, vy, 300, 32, 6, (17, 60, 30))
    c.draw_text(62, vy + 8, "VERIFIED DETERMINISTIC", GREEN, 2)
    c.draw_text(380, vy + 8, "2 runs, identical output", DIM, 2)

    # Feature callouts
    fy = 590
    c.draw_hline(40, fy, W - 80, DIM, 1)
    fy += 16

    features = [
        ("Kahan summation", "Zero rounding drift on 1M+ rows"),
        ("Welford variance", "Single-pass streaming, O(1) memory"),
        ("--verify flag", "Runs twice, proves identical output"),
        ("CSV / TSV / JSONL", "Auto-detects format from extension"),
    ]
    for feat, desc in features:
        c.draw_text(50, fy, feat, ACCENT, 2)
        c.draw_text(380, fy, desc, DIM, 2)
        fy += 28

    # Bottom bar
    c.fill_rect(0, H - 50, W, 50, TITLE_BG)
    c.draw_text_centered(H - 38, "CJC-Lang v0.1.4  |  cargo install cjc-lang  |  adamezzat.dev", DIM, 2)

    # Emoji indicator top-right
    c.draw_text(W - 200, 22, "Streaming", YELLOW, 2)

    c.save(os.path.join(OUTPUT_DIR, "cmd_01_flow.png"))


def generate_schema():
    """Generate schema command slide."""
    c = Canvas()

    for py in range(40):
        alpha = 1.0 - py / 40.0
        r = int(BG[0] + (CYAN[0] - BG[0]) * alpha * 0.3)
        g = int(BG[1] + (CYAN[1] - BG[1]) * alpha * 0.3)
        b = int(BG[2] + (CYAN[2] - BG[2]) * alpha * 0.3)
        c.fill_rect(0, py, W, 1, (r, g, b))

    c.draw_text(40, 16, "cjcl schema", CYAN, 3)
    c.draw_text(360, 22, "Auto-Detect Data Types", DIM, 2)

    # Command
    ty = draw_terminal_frame(c, 30, 60, W - 60, 100, "Terminal")
    c.draw_text(50, ty, "$ cjcl schema users.csv --full --save schema.json", GREEN, 2)

    # Output
    oy = draw_terminal_frame(c, 30, 175, W - 60, 340, "Output")
    c.draw_text(50, oy, "Schema for users.csv (50,000 rows sampled):", WHITE, 2)
    oy += 30

    headers = ["Column", "Type", "Nulls", "Unique", "Sample"]
    rows = [
        ["user_id",    "integer", "0",     "50000", "1, 2, 3"],
        ["name",       "string",  "12",    "49231", "Alice, Bob"],
        ["email",      "string",  "0",     "50000", "a@b.com"],
        ["age",        "float",   "847",   "89",    "25.0, 31.5"],
        ["signup",     "string",  "0",     "1247",  "2024-01-15"],
        ["score",      "float",   "2103",  "4521",  "0.85, 0.92"],
    ]
    type_colors = {
        "integer": ACCENT, "string": YELLOW, "float": ORANGE
    }
    for ri, row in enumerate(rows):
        cx = 50
        for ci, cell in enumerate(row):
            if ci == 0:
                color = WHITE
            elif ci == 1:
                color = type_colors.get(cell, WHITE)
            elif ci == 2:
                color = RED if int(cell) > 0 else GREEN
            else:
                color = DIM
            c.draw_text(cx, oy, cell, color, 2)
            cx += [12, 10, 8, 9, 16][ci] * 12
        oy += 22

    # Schema save confirmation
    oy += 10
    c.draw_text(50, oy, "Schema saved to schema.json", GREEN, 2)

    # CI gating section
    gy = 540
    ty2 = draw_terminal_frame(c, 30, gy, W - 60, 130, "CI Pipeline")
    c.draw_text(50, ty2, "$ cjcl schema users_v2.csv --check schema.json --strict", GREEN, 2)
    c.draw_text(50, ty2 + 28, "SCHEMA MISMATCH:", RED, 2)
    c.draw_text(50, ty2 + 52, "  + new column: 'phone' (string)", YELLOW, 2)
    c.draw_text(50, ty2 + 74, "  ~ type change: 'age' integer -> float", ORANGE, 2)

    # Features
    fy = 710
    c.draw_hline(40, fy, W - 80, DIM, 1)
    fy += 16
    features = [
        ("--save / --check", "Save schema, enforce in CI"),
        ("--diff", "Show +/- changes between versions"),
        ("--full", "Type distributions + sample values"),
        ("Parquet/Arrow/SQLite", "Reads metadata without deps"),
    ]
    for feat, desc in features:
        c.draw_text(50, fy, feat, CYAN, 2)
        c.draw_text(380, fy, desc, DIM, 2)
        fy += 28

    c.fill_rect(0, H - 50, W, 50, TITLE_BG)
    c.draw_text_centered(H - 38, "CJC-Lang v0.1.4  |  cargo install cjc-lang  |  adamezzat.dev", DIM, 2)

    c.save(os.path.join(OUTPUT_DIR, "cmd_02_schema.png"))


def generate_inspect():
    """Generate inspect command slide."""
    c = Canvas()

    for py in range(40):
        alpha = 1.0 - py / 40.0
        r = int(BG[0] + (GREEN[0] - BG[0]) * alpha * 0.3)
        g = int(BG[1] + (GREEN[1] - BG[1]) * alpha * 0.3)
        b = int(BG[2] + (GREEN[2] - BG[2]) * alpha * 0.3)
        c.fill_rect(0, py, W, 1, (r, g, b))

    c.draw_text(40, 16, "cjcl inspect", GREEN, 3)
    c.draw_text(400, 22, "Understand Any File", DIM, 2)

    # CSV inspection
    ty = draw_terminal_frame(c, 30, 60, W - 60, 280, "Inspect CSV")
    c.draw_text(50, ty, "$ cjcl inspect dataset.csv --deep", GREEN, 2)
    ty += 28
    props = [
        ("File:", "dataset.csv", WHITE),
        ("Type:", "CSV (comma-separated)", CYAN),
        ("Rows:", "1,000,000", ORANGE),
        ("Columns:", "8", ORANGE),
        ("Size:", "142.3 MB", ORANGE),
        ("SHA-256:", "a7f3c2...9e1b04", DIM),
    ]
    for label, val, color in props:
        c.draw_text(50, ty, label, DIM, 2)
        c.draw_text(200, ty, val, color, 2)
        ty += 22

    ty += 6
    c.draw_text(50, ty, "Column Stats:", ACCENT, 2)
    ty += 24
    headers = ["Column", "Type", "Nulls", "Mean", "Std", "Unique"]
    stats = [
        ["price",  "float", "0",   "29.95", "12.41", "8921"],
        ["qty",    "int",   "45",  "3.2",   "2.1",   "150"],
    ]
    for row in stats:
        cx = 50
        for ci, cell in enumerate(row):
            color = [WHITE, CYAN, RED if cell != "0" else GREEN, ORANGE, ORANGE, DIM][ci]
            c.draw_text(cx, ty, cell, color, 2)
            cx += [11, 8, 8, 9, 9, 10][ci] * 12
        ty += 22

    # Model file inspection
    my = draw_terminal_frame(c, 30, 365, W - 60, 210, "Inspect Model (Safe)")
    c.draw_text(50, my, "$ cjcl inspect model.onnx --hash", GREEN, 2)
    my += 28
    mprops = [
        ("File:", "model.onnx", WHITE),
        ("Type:", "ONNX model (safe metadata only)", YELLOW),
        ("Size:", "847.2 MB", ORANGE),
        ("Magic:", "08 00 12 04 6F 6E 6E 78", DIM),
        ("SHA-256:", "e4c912...7b3f01", DIM),
    ]
    for label, val, color in mprops:
        c.draw_text(50, my, label, DIM, 2)
        c.draw_text(200, my, val, color, 2)
        my += 22

    my += 4
    c.fill_rounded_rect(50, my, 580, 28, 4, (60, 40, 10))
    c.draw_text(62, my + 6, "Model never executed. Metadata only.", YELLOW, 2)

    # Features
    fy = 600
    c.draw_hline(40, fy, W - 80, DIM, 1)
    fy += 16
    features = [
        (".cjc source", "Parses + shows AST stats"),
        (".csv/.tsv/.jsonl", "Full column statistics"),
        (".pkl/.onnx/.joblib", "Safe metadata (never run)"),
        ("--compare", "Side-by-side file comparison"),
        ("--manifest", "Machine-readable: hash size type path"),
    ]
    for feat, desc in features:
        c.draw_text(50, fy, feat, GREEN, 2)
        c.draw_text(370, fy, desc, DIM, 2)
        fy += 26

    c.fill_rect(0, H - 50, W, 50, TITLE_BG)
    c.draw_text_centered(H - 38, "CJC-Lang v0.1.4  |  cargo install cjc-lang  |  adamezzat.dev", DIM, 2)

    c.save(os.path.join(OUTPUT_DIR, "cmd_03_inspect.png"))


def generate_patch():
    """Generate patch command slide."""
    c = Canvas()

    for py in range(40):
        alpha = 1.0 - py / 40.0
        r = int(BG[0] + (YELLOW[0] - BG[0]) * alpha * 0.3)
        g = int(BG[1] + (YELLOW[1] - BG[1]) * alpha * 0.3)
        b = int(BG[2] + (YELLOW[2] - BG[2]) * alpha * 0.3)
        c.fill_rect(0, py, W, 1, (r, g, b))

    c.draw_text(40, 16, "cjcl patch", YELLOW, 3)
    c.draw_text(320, 22, "Clean Data Without Code", DIM, 2)

    # Command
    ty = draw_terminal_frame(c, 30, 60, W - 60, 170, "Terminal")
    c.draw_text(50, ty, "$ cjcl patch messy_data.csv \\", GREEN, 2)
    c.draw_text(50, ty + 24, "    --nan-fill 0 \\", GREEN, 2)
    c.draw_text(50, ty + 48, "    --drop temp_col \\", GREEN, 2)
    c.draw_text(50, ty + 72, "    --rename old_name new_name \\", GREEN, 2)
    c.draw_text(50, ty + 96, "    --dry-run", CYAN, 2)

    # Dry run output
    oy = draw_terminal_frame(c, 30, 250, W - 60, 250, "Dry Run Preview")
    c.draw_text(50, oy, "Patch Plan:", ACCENT, 2)
    oy += 26
    c.draw_text(50, oy, "  source:", DIM, 2)
    c.draw_text(200, oy, "messy_data.csv", WHITE, 2)
    oy += 22
    c.draw_text(50, oy, "  format:", DIM, 2)
    c.draw_text(200, oy, "CSV", CYAN, 2)
    oy += 22
    c.draw_text(50, oy, "  transforms (3):", DIM, 2)
    oy += 24
    transforms = [
        ("1.", "nan-fill: replace NaN/NA/null with '0'", ORANGE),
        ("2.", "drop column: 'temp_col'", RED),
        ("3.", "rename: 'old_name' -> 'new_name'", YELLOW),
    ]
    for num, desc, color in transforms:
        c.draw_text(70, oy, num, DIM, 2)
        c.draw_text(110, oy, desc, color, 2)
        oy += 24

    oy += 10
    c.fill_rounded_rect(50, oy, 430, 28, 4, (40, 50, 20))
    c.draw_text(62, oy + 6, "DRY RUN: no changes written to disk", GREEN, 2)

    # Actual run
    ay = draw_terminal_frame(c, 30, 520, W - 60, 110, "Apply")
    c.draw_text(50, ay, "$ cjcl patch messy_data.csv --nan-fill 0 \\", GREEN, 2)
    c.draw_text(50, ay + 24, "    --drop temp_col --in-place --backup", GREEN, 2)
    c.draw_text(50, ay + 56, "patched 1,000,000 rows", WHITE, 2)
    c.draw_text(380, ay + 56, "(backup: messy_data.csv.bak)", DIM, 2)

    # Features
    fy = 660
    c.draw_hline(40, fy, W - 80, DIM, 1)
    fy += 16
    features = [
        ("--nan-fill", "Replace all NaN/NA/null values"),
        ("--impute", "Fill NaN with column mean"),
        ("--drop / --rename", "Remove or rename columns"),
        ("--dry-run", "Preview before applying"),
        ("--backup", "Auto-save .bak before overwrite"),
        ("--check", "Validate transforms (CI gating)"),
    ]
    for feat, desc in features:
        c.draw_text(50, fy, feat, YELLOW, 2)
        c.draw_text(370, fy, desc, DIM, 2)
        fy += 26

    c.fill_rect(0, H - 50, W, 50, TITLE_BG)
    c.draw_text_centered(H - 38, "CJC-Lang v0.1.4  |  cargo install cjc-lang  |  adamezzat.dev", DIM, 2)

    c.save(os.path.join(OUTPUT_DIR, "cmd_04_patch.png"))


def generate_drift():
    """Generate drift command slide."""
    c = Canvas()

    for py in range(40):
        alpha = 1.0 - py / 40.0
        r = int(BG[0] + (RED[0] - BG[0]) * alpha * 0.3)
        g = int(BG[1] + (RED[1] - BG[1]) * alpha * 0.3)
        b = int(BG[2] + (RED[2] - BG[2]) * alpha * 0.3)
        c.fill_rect(0, py, W, 1, (r, g, b))

    c.draw_text(40, 16, "cjcl drift", RED, 3)
    c.draw_text(310, 22, "Math-Aware Data Diff", DIM, 2)

    # Command
    ty = draw_terminal_frame(c, 30, 60, W - 60, 80, "Terminal")
    c.draw_text(50, ty, "$ cjcl drift baseline.csv updated.csv --tolerance 0.001", GREEN, 2)

    # Summary output
    oy = draw_terminal_frame(c, 30, 155, W - 60, 280, "Drift Report")
    c.draw_text(50, oy, "Drift Report:", ACCENT, 2)
    oy += 28

    metrics = [
        ("Rows (A):", "1,000,000", WHITE),
        ("Rows (B):", "1,000,000", WHITE),
        ("Schema match:", "true", GREEN),
        ("Cell differences:", "3,847", ORANGE),
        ("NaN divergences:", "0", GREEN),
        ("Max deviation:", "0.000892", YELLOW),
        ("Mean deviation:", "0.000034", GREEN),
        ("Frobenius norm:", "0.0412", ORANGE),
    ]
    for label, val, color in metrics:
        c.draw_text(50, oy, label, DIM, 2)
        c.draw_text(310, oy, val, color, 2)
        oy += 24

    oy += 8
    c.draw_text(50, oy, "First differences:", WHITE, 2)
    oy += 24
    diffs = [
        ('[42,revenue]:', '"29.9501"', '"29.9493"'),
        ('[108,tax]:',    '"0.0850"',  '"0.0851"'),
        ('[250,score]:',  '"0.8821"',  '"0.8829"'),
    ]
    for loc, a, b in diffs:
        c.draw_text(70, oy, loc, DIM, 2)
        c.draw_text(280, oy, a, RED, 2)
        c.draw_text(430, oy, "->", DIM, 2)
        c.draw_text(480, oy, b, GREEN, 2)
        oy += 22

    # CI gating
    gy = draw_terminal_frame(c, 30, 545, W - 60, 100, "CI Gate")
    c.draw_text(50, gy, "$ cjcl drift v1.csv v2.csv --fail-on-diff", GREEN, 2)
    c.draw_text(50, gy + 28, "DRIFT DETECTED: 3847 differences", RED, 2)
    c.draw_text(50, gy + 52, "Exit code: 1", RED, 2)

    # Features
    fy = 670
    c.draw_hline(40, fy, W - 80, DIM, 1)
    fy += 16
    features = [
        ("--tolerance", "Set acceptable deviation threshold"),
        ("--fail-on-diff", "CI gate: exit 1 on any change"),
        ("Frobenius norm", "Overall dataset distance metric"),
        ("--report <file>", "Save full diff as JSON"),
        ("CSV / JSONL / text", "Auto-detect comparison mode"),
    ]
    for feat, desc in features:
        c.draw_text(50, fy, feat, RED, 2)
        c.draw_text(370, fy, desc, DIM, 2)
        fy += 26

    c.fill_rect(0, H - 50, W, 50, TITLE_BG)
    c.draw_text_centered(H - 38, "CJC-Lang v0.1.4  |  cargo install cjc-lang  |  adamezzat.dev", DIM, 2)

    c.save(os.path.join(OUTPUT_DIR, "cmd_05_drift.png"))


def generate_seek():
    """Generate seek command slide."""
    c = Canvas()

    for py in range(40):
        alpha = 1.0 - py / 40.0
        r = int(BG[0] + (PURPLE[0] - BG[0]) * alpha * 0.3)
        g = int(BG[1] + (PURPLE[1] - BG[1]) * alpha * 0.3)
        b = int(BG[2] + (PURPLE[2] - BG[2]) * alpha * 0.3)
        c.fill_rect(0, py, W, 1, (r, g, b))

    c.draw_text(40, 16, "cjcl seek", PURPLE, 3)
    c.draw_text(280, 22, "Deterministic File Discovery", DIM, 2)

    # Basic search
    ty = draw_terminal_frame(c, 30, 60, W - 60, 230, "Find Files")
    c.draw_text(50, ty, "$ cjcl seek ./project --type cjcl --hash", GREEN, 2)
    ty += 28

    files = [
        ("a1b2c3d4e5f6a7b8", "src/main.cjc", GREEN),
        ("b2c3d4e5f6a7b8c9", "src/math/linalg.cjc", GREEN),
        ("c3d4e5f6a7b8c9d0", "src/stats/regression.cjc", GREEN),
        ("d4e5f6a7b8c9d0e1", "tests/test_flow.cjc", GREEN),
        ("e5f6a7b8c9d0e1f2", "tests/test_schema.cjc", GREEN),
    ]
    for hash_p, path, color in files:
        c.draw_text(50, ty, hash_p, DIM, 2)
        c.draw_text(280, ty, path, color, 2)
        ty += 22

    ty += 8
    c.draw_text(50, ty, "5 files found", ACCENT, 2)

    # Content search
    cy = draw_terminal_frame(c, 30, 310, W - 60, 180, "Content Search")
    c.draw_text(50, cy, "$ cjcl seek . --contains \"tensor\" --type cjcl --first 3", GREEN, 2)
    cy += 28

    results = [
        ("src/ml/train.cjcl", "12:", "let weights: Tensor = zeros(784, 128);"),
        ("src/ml/loss.cjc", "5:", "fn mse(pred: Tensor, target: Tensor) -> f64 {"),
        ("src/ad/grad.cjc", "8:", "let grad: Tensor = backward(loss);"),
    ]
    for path, line, content in results:
        c.draw_text(50, cy, path, YELLOW, 2)
        c.draw_text(50, cy + 20, line, DIM, 2)
        c.draw_text(90, cy + 20, content, WHITE, 2)
        cy += 44

    # Manifest mode
    my = draw_terminal_frame(c, 30, 510, W - 60, 110, "Manifest Mode")
    c.draw_text(50, my, "$ cjcl seek data/ --type csv --manifest", GREEN, 2)
    my += 28
    c.draw_text(50, my, "a7f3c2..9e1b 142300000 data/sales.csv", CYAN, 2)
    c.draw_text(50, my + 22, "b8e4d1..3f2a  89100000 data/users.csv", CYAN, 2)
    c.draw_text(50, my + 44, "c9f5e2..4g3b  23400000 data/logs.csv", CYAN, 2)

    # Features
    fy = 650
    c.draw_hline(40, fy, W - 80, DIM, 1)
    fy += 16
    features = [
        ("--contains", "Search inside file contents"),
        ("--type / --min-size", "Filter by ext, size"),
        ("--hash / --manifest", "SHA-256 per file, machine output"),
        ("--sort name|size|mod", "Deterministic sort order"),
        ("Always sorted", "Lexicographic, reproducible"),
    ]
    for feat, desc in features:
        c.draw_text(50, fy, feat, PURPLE, 2)
        c.draw_text(370, fy, desc, DIM, 2)
        fy += 26

    c.fill_rect(0, H - 50, W, 50, TITLE_BG)
    c.draw_text_centered(H - 38, "CJC-Lang v0.1.4  |  cargo install cjc-lang  |  adamezzat.dev", DIM, 2)

    c.save(os.path.join(OUTPUT_DIR, "cmd_06_seek.png"))


def generate_forge():
    """Generate forge command slide."""
    c = Canvas()

    for py in range(40):
        alpha = 1.0 - py / 40.0
        r = int(BG[0] + (MAGENTA[0] - BG[0]) * alpha * 0.3)
        g = int(BG[1] + (MAGENTA[1] - BG[1]) * alpha * 0.3)
        b = int(BG[2] + (MAGENTA[2] - BG[2]) * alpha * 0.3)
        c.fill_rect(0, py, W, 1, (r, g, b))

    c.draw_text(40, 16, "cjcl forge", MAGENTA, 3)
    c.draw_text(320, 22, "Reproducibility Enforced", DIM, 2)

    # Forge run
    ty = draw_terminal_frame(c, 30, 60, W - 60, 220, "Forge Run")
    c.draw_text(50, ty, "$ cjcl forge run train.cjcl --seed 42", GREEN, 2)
    ty += 28

    props = [
        ("Source:", "train.cjcl", WHITE),
        ("Seed:", "42", ORANGE),
        ("Output lines:", "847", ORANGE),
        ("Output bytes:", "24,891", ORANGE),
        ("SHA-256:", "e4c912a7f3b2d1...8c9e7b3f01", CYAN),
        ("Artifact:", ".cjc-forge/e4c912a7f3b2d1.txt", DIM),
    ]
    for label, val, color in props:
        c.draw_text(50, ty, label, DIM, 2)
        c.draw_text(250, ty, val, color, 2)
        ty += 22

    ty += 8
    c.fill_rounded_rect(50, ty, 160, 28, 4, (17, 60, 30))
    c.draw_text(62, ty + 6, "FORGED", GREEN, 2)

    # Verify
    vy = draw_terminal_frame(c, 30, 310, W - 60, 170, "Verify Later")
    c.draw_text(50, vy, "$ cjcl forge verify train.cjcl e4c912a7f3b2d1", GREEN, 2)
    vy += 28
    c.draw_text(50, vy, "Verifying train.cjcl...", WHITE, 2)
    vy += 24
    c.draw_text(50, vy, "Expected:", DIM, 2)
    c.draw_text(200, vy, "e4c912a7f3b2d1...8c9e7b3f01", CYAN, 2)
    vy += 22
    c.draw_text(50, vy, "Actual:", DIM, 2)
    c.draw_text(200, vy, "e4c912a7f3b2d1...8c9e7b3f01", CYAN, 2)
    vy += 24
    c.fill_rounded_rect(50, vy, 120, 28, 4, (17, 60, 30))
    c.draw_text(62, vy + 6, "MATCH", GREEN, 2)

    # List artifacts
    ly = draw_terminal_frame(c, 30, 500, W - 60, 150, "List Artifacts")
    c.draw_text(50, ly, "$ cjcl forge list", GREEN, 2)
    ly += 28
    arts = [
        ("e4c912a7f3b2d1", "train.cjcl", "42", "847", "24.3 KB"),
        ("a1b2c3d4e5f6a7", "eval.cjcl",  "42", "128", "3.1 KB"),
        ("f8e7d6c5b4a3f2", "bench.cjcl", "99", "2041", "89.7 KB"),
    ]
    # Header
    cx_starts = [50, 250, 410, 480, 560]
    hdrs = ["Hash", "Source", "Seed", "Lines", "Size"]
    for i, h in enumerate(hdrs):
        c.draw_text(cx_starts[i], ly, h, ACCENT, 2)
    ly += 22
    c.draw_hline(50, ly, 650, DIM, 1)
    ly += 6
    for hash_s, src, seed, lines, size in arts:
        c.draw_text(50, ly, hash_s, DIM, 2)
        c.draw_text(250, ly, src, MAGENTA, 2)
        c.draw_text(410, ly, seed, ORANGE, 2)
        c.draw_text(480, ly, lines, WHITE, 2)
        c.draw_text(560, ly, size, WHITE, 2)
        ly += 22

    ly += 4
    c.draw_text(50, ly, "3 artifacts", DIM, 2)

    # Features
    fy = 680
    c.draw_hline(40, fy, W - 80, DIM, 1)
    fy += 16
    features = [
        ("Content-addressed", "Output stored by SHA-256 hash"),
        ("forge verify", "Prove output hasn't changed"),
        ("forge list/show", "Browse cached artifacts"),
        ("--seed", "Control deterministic RNG"),
    ]
    for feat, desc in features:
        c.draw_text(50, fy, feat, MAGENTA, 2)
        c.draw_text(370, fy, desc, DIM, 2)
        fy += 26

    c.fill_rect(0, H - 50, W, 50, TITLE_BG)
    c.draw_text_centered(H - 38, "CJC-Lang v0.1.4  |  cargo install cjc-lang  |  adamezzat.dev", DIM, 2)

    c.save(os.path.join(OUTPUT_DIR, "cmd_07_forge.png"))


if __name__ == "__main__":
    print("Generating CJC CLI command slides...")
    generate_flow()
    generate_schema()
    generate_inspect()
    generate_patch()
    generate_drift()
    generate_seek()
    generate_forge()
    print("\nDone! 7 slides generated.")
