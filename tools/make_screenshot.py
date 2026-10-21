#!/usr/bin/env python3
"""
make_screenshot.py — Render benchmark terminal output as a PNG screenshot.

Usage:
    python3 tools/make_screenshot.py \
        --input benchmark_results/pz_bf16_t4.txt \
        --output benchmark_results/screenshots/pz_bf16_t4.png \
        --title "Project Zero · BF16 · 4 threads" \
        --command "./pz --model models/bitnet-b1.58-2B-4T.bin ..."
"""

import argparse
import os
import re
import sys
from PIL import Image, ImageDraw, ImageFont

# Terminal colour palette (dark theme)
BG        = (15,  15,  20)      # near-black background
FG        = (204, 204, 204)     # default text
GREEN     = (78,  200, 78)      # prompt / highlights
CYAN      = (86,  218, 220)     # header / labels
YELLOW    = (230, 200, 50)      # tok/s value
WHITE     = (255, 255, 255)     # output text
DIM       = (110, 110, 120)     # separator lines
ORANGE    = (230, 140, 30)      # command text
TITLE_BG  = (30,  30,  40)      # title bar bg
PADDING   = 28
LINE_H    = 22
FONT_SIZE = 16
IMG_WIDTH = 900

def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def colour_line(line: str):
    """Return (colour, line_text) based on content."""
    if re.search(r'\[gen\].*tok/s', line):
        return YELLOW, line
    if line.startswith('Active:') or line.startswith('SIMD:') or line.startswith('Classifier:'):
        return CYAN, line
    if '│' in line or '├' in line or '┌' in line or '└' in line:
        return CYAN, line
    if line.startswith('  Response:') or line.startswith(' ') and len(line) > 5 and not line.strip().startswith('│'):
        return WHITE, line
    if line.startswith('Project Zero Engine') or line.startswith('Model format'):
        return GREEN, line
    if line.startswith('Model Configuration'):
        return DIM, line
    if re.match(r'\s+(dim|hidden_dim|n_layers|n_heads|vocab_size|seq_len|act_type|rope_theta|head_dim|kv_dim):', line):
        return DIM, line
    if 'KV Strategy' in line or line.startswith('KV cache:'):
        return DIM, line
    return FG, line


def render(text: str, title: str, command: str, output_path: str):
    font = load_font(FONT_SIZE)
    bold_font = load_font(FONT_SIZE)  # same size; bold not critical

    lines = text.splitlines()
    # Filter out very verbose model-config lines to keep screenshot compact
    skip_patterns = [
        r'^\s+(dim|hidden_dim|n_layers|n_heads|n_kv_heads|vocab_size|seq_len|act_type|rope_theta|head_dim|kv_dim):',
        r'^KV Strategy:',
        r'^KV cache:',
        r'^Model format:',
    ]
    filtered = []
    for line in lines:
        skip = any(re.match(p, line) for p in skip_patterns)
        if not skip:
            filtered.append(line)

    # Prepend command line
    display_lines = []
    display_lines.append((GREEN, '$ ' + command))
    display_lines.append((DIM, '─' * 90))
    for line in filtered:
        col, txt = colour_line(line)
        display_lines.append((col, txt))

    n_lines = len(display_lines)
    img_height = PADDING * 2 + 36 + n_lines * LINE_H + 10  # 36 for title bar

    img = Image.new('RGB', (IMG_WIDTH, img_height), BG)
    draw = ImageDraw.Draw(img)

    # Title bar
    draw.rectangle([0, 0, IMG_WIDTH, 36], fill=TITLE_BG)
    draw.text((PADDING, 8), title, font=bold_font, fill=WHITE)

    # Lines
    y = 36 + PADDING
    for col, text in display_lines:
        # Truncate to fit width
        max_chars = (IMG_WIDTH - 2 * PADDING) // (FONT_SIZE // 2)
        if len(text) > max_chars:
            text = text[:max_chars - 3] + '...'
        draw.text((PADDING, y), text, font=font, fill=col)
        y += LINE_H

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    img.save(output_path)
    print(f'Screenshot saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Render benchmark output as terminal PNG')
    parser.add_argument('--input',   required=True, help='Input benchmark .txt file')
    parser.add_argument('--output',  required=True, help='Output PNG path')
    parser.add_argument('--title',   required=True, help='Title bar text')
    parser.add_argument('--command', required=True, help='Command that was run')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        text = f.read()

    render(text, args.title, args.command, args.output)


if __name__ == '__main__':
    main()
