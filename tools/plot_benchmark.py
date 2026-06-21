#!/usr/bin/env python3
"""
plot_benchmark.py — Generate comparison graphs from benchmark results.

Outputs:
  benchmark_results/comparison_graph.png  — tok/s vs threads, 3 series
  benchmark_results/speedup_graph.png     — PZ speedup over MSFT
  benchmark_results/bar_comparison.png    — bar chart at peak threads
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Measured data (i5-11300H, AVX-512 VNNI, 2026-06-21) ──────────────────────
threads   = [1, 2, 3, 4, 5, 6, 7, 8]

pz_bf16   = [12.17, 23.27, 23.94, 25.12, 24.95, 25.00, 28.11, 22.16]
pz_int4   = [20.15, 32.80, 39.89, 42.76, 37.57, 42.83, 35.05, 32.71]
msft      = [ 2.11,  3.83,  5.31,  6.73,  5.32,  6.29,  6.60,  6.10]

# Peak values
pk_bf16 = max(pz_bf16)
pk_int4 = max(pz_int4)
pk_msft = max(msft)

# Colours
C_BF16  = '#4EC84E'   # green
C_INT4  = '#00BFFF'   # cyan-blue
C_MSFT  = '#FF6B35'   # orange-red
BG      = '#0F0F14'
GRID    = '#2A2A35'
FG      = '#CCCCCC'

def apply_dark(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.spines['bottom'].set_color(GRID)
    ax.spines['top'].set_color(GRID)
    ax.spines['left'].set_color(GRID)
    ax.spines['right'].set_color(GRID)
    ax.grid(color=GRID, linestyle='--', linewidth=0.7)


# ── Graph 1: tok/s vs threads ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
apply_dark(ax, fig)

ax.plot(threads, pz_int4, 'o-', color=C_INT4, lw=2.5, ms=7, label='Project Zero INT4')
ax.plot(threads, pz_bf16, 's-', color=C_BF16, lw=2.5, ms=7, label='Project Zero BF16')
ax.plot(threads, msft,    '^-', color=C_MSFT, lw=2.5, ms=7, label='Microsoft bitnet.cpp (i2_s)')

ax.set_xlabel('Threads', fontsize=13)
ax.set_ylabel('Tokens / second', fontsize=13)
ax.set_title('BitNet b1.58-2B-4T · Token Throughput vs Thread Count\n'
             'i5-11300H @ 3.10 GHz · AVX-512 VNNI · 16 GB DDR4', fontsize=13)
ax.set_xticks(threads)
ax.legend(facecolor='#1A1A25', edgecolor=GRID, labelcolor=FG, fontsize=11)

# Annotate peaks
ax.annotate(f'{pk_int4:.1f} tok/s',
            xy=(pz_int4.index(pk_int4)+1, pk_int4),
            xytext=(5, 8), textcoords='offset points',
            color=C_INT4, fontsize=10, fontweight='bold')
ax.annotate(f'{pk_bf16:.2f} tok/s',
            xy=(pz_bf16.index(pk_bf16)+1, pk_bf16),
            xytext=(5, -18), textcoords='offset points',
            color=C_BF16, fontsize=10, fontweight='bold')
ax.annotate(f'{pk_msft:.2f} tok/s',
            xy=(msft.index(pk_msft)+1, pk_msft),
            xytext=(5, 5), textcoords='offset points',
            color=C_MSFT, fontsize=10, fontweight='bold')

out = 'benchmark_results/comparison_graph.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout()
plt.savefig(out, dpi=150, facecolor=BG)
plt.close()
print(f'Saved: {out}')


# ── Graph 2: Speedup ratio ────────────────────────────────────────────────────
speedup_bf16 = [b/m for b, m in zip(pz_bf16, msft)]
speedup_int4 = [i/m for i, m in zip(pz_int4, msft)]

fig, ax = plt.subplots(figsize=(10, 5))
apply_dark(ax, fig)

ax.plot(threads, speedup_int4, 'o-', color=C_INT4, lw=2.5, ms=7, label='PZ INT4 vs MSFT')
ax.plot(threads, speedup_bf16, 's-', color=C_BF16, lw=2.5, ms=7, label='PZ BF16 vs MSFT')
ax.axhline(1.0, color=C_MSFT, lw=1.5, linestyle='--', label='Microsoft baseline (1×)')

ax.set_xlabel('Threads', fontsize=13)
ax.set_ylabel('Speedup (×)', fontsize=13)
ax.set_title('Project Zero Speedup over Microsoft bitnet.cpp\n'
             'i5-11300H · Same model (BitNet b1.58-2B-4T) · Same prompt · 40 tokens', fontsize=13)
ax.set_xticks(threads)
ax.legend(facecolor='#1A1A25', edgecolor=GRID, labelcolor=FG, fontsize=11)

for t, su in zip(threads, speedup_int4):
    ax.annotate(f'{su:.1f}×', xy=(t, su), xytext=(0, 6),
                textcoords='offset points', ha='center',
                color=C_INT4, fontsize=9)

out = 'benchmark_results/speedup_graph.png'
plt.tight_layout()
plt.savefig(out, dpi=150, facecolor=BG)
plt.close()
print(f'Saved: {out}')


# ── Graph 3: Bar chart at best thread count ───────────────────────────────────
# Use t=4 (physical core count) for a clean, fair comparison
t4_idx = threads.index(4)
t_best_pz_int4 = pz_int4.index(pk_int4) + 1
labels   = ['PZ INT4\n(best)', 'PZ BF16\n(best)', 'MSFT\nbitnet.cpp']
values   = [pk_int4,  pk_bf16,  pk_msft]
colours  = [C_INT4, C_BF16, C_MSFT]
t_labels = [f't={t_best_pz_int4}', f't={pz_bf16.index(pk_bf16)+1}', f't={msft.index(pk_msft)+1}']

fig, ax = plt.subplots(figsize=(8, 5))
apply_dark(ax, fig)

bars = ax.bar(labels, values, color=colours, width=0.45, edgecolor=GRID, linewidth=0.8)
for bar, val, tlabel in zip(bars, values, t_labels):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f} tok/s\n({tlabel})',
            ha='center', va='bottom', color=FG, fontsize=11, fontweight='bold')

ax.set_ylabel('Tokens / second', fontsize=13)
ax.set_title('Peak Throughput — Project Zero vs Microsoft bitnet.cpp\n'
             'BitNet b1.58-2B-4T · i5-11300H @ 3.10 GHz · AVX-512 VNNI', fontsize=13)
ax.set_ylim(0, pk_int4 * 1.25)

# Speedup annotations — as figure text below the chart
speedup_int4 = pk_int4 / pk_msft
speedup_bf16 = pk_bf16 / pk_msft
plt.figtext(0.26, 0.01, f'{speedup_int4:.1f}× faster than MSFT',
            ha='center', color=C_INT4, fontsize=10, style='italic')
plt.figtext(0.57, 0.01, f'{speedup_bf16:.1f}× faster than MSFT',
            ha='center', color=C_BF16, fontsize=10, style='italic')

out = 'benchmark_results/bar_comparison.png'
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(out, dpi=150, facecolor=BG)
plt.close()
print(f'Saved: {out}')

print('\nAll graphs generated successfully.')
print(f'\nKey results (i5-11300H, 2026-06-21):')
print(f'  PZ INT4 peak  : {pk_int4:.2f} tok/s')
print(f'  PZ BF16 peak  : {pk_bf16:.2f} tok/s')
print(f'  MSFT peak     : {pk_msft:.2f} tok/s')
print(f'  PZ INT4 / MSFT: {pk_int4/pk_msft:.2f}×')
print(f'  PZ BF16 / MSFT: {pk_bf16/pk_msft:.2f}×')
