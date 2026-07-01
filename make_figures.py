"""Generate professional, self-explanatory figures for the README / showcase.

Plain-English labels (no internal codenames), white background (renders on GitHub light+dark), value
labels, chance reference lines. Outputs to figures/. Run: python3 make_figures.py
"""
import os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
C_OLD = "#4C72B0"      # backprop-free model (the constrained system)
C_NEW = "#DD8452"      # standard model with backprop (the control)
C_CH = "#9AA0A6"       # chance / reference


# ---------- Figure 1: headline capability comparison ----------
def fig_capability():
    tasks = ["Understand relations\n(synthetic)", "Chain 2 steps\n(synthetic)", "Understand relations\n(REAL data, SNLI)"]
    old = [65.7, 20.0, 33.4]          # backprop-free model (best it can do)
    new = [100.0, 100.0, 69.97]       # standard model with backprop
    chance = [33.3, 20.0, 33.4]
    x = range(len(tasks)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2))
    from matplotlib.lines import Line2D
    b1 = ax.bar([i - w/2 for i in x], old, w, label="Backprop-free model (4B, memory-efficient)", color=C_OLD)
    b2 = ax.bar([i + w/2 for i in x], new, w, label="Standard model, with backprop (control)", color=C_NEW)
    for i, c in enumerate(chance):
        ax.plot([i - w*1.15, i + w*1.15], [c, c], color=C_CH, ls="--", lw=1.5, zorder=0)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.5, f"{r.get_height():.0f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(tasks); ax.set_ylim(0, 112); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Same tasks, two model designs — the limit is the architecture, not the task",
                 fontsize=12.5, fontweight="bold", pad=14)
    handles = [b1, b2, Line2D([0], [0], color=C_CH, ls="--", lw=1.5, label="random guessing (chance level)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "capability_comparison.png"), dpi=150); plt.close(fig)


# ---------- Figure 2: the multi-step "reasoning depth" wall ----------
def fig_depth():
    k = [2, 3, 4, 6, 8]
    new = [100, 100, 21.3, 19.8, 21.4]     # standard model with backprop
    old = [20, 20, 20, 20, 20]             # backprop-free model: chance at every depth (fails even k=2)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axhspan(0, 25, color=C_CH, alpha=0.12); ax.text(8, 12, "random-guessing zone", ha="right", color=C_CH, fontsize=9)
    ax.plot(k, new, "-o", color=C_NEW, lw=2.4, ms=8, label="Standard model, with backprop")
    ax.plot(k, old, "-s", color=C_OLD, lw=2.4, ms=7, label="Backprop-free model")
    for xi, yi in zip(k, new):
        ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9.5, fontweight="bold")
    ax.annotate("solves 2–3 steps", (2.5, 100), textcoords="offset points", xytext=(0, -22),
                ha="center", color=C_NEW, fontsize=10, fontweight="bold")
    ax.annotate("a wall — 5× bigger model\n& more training didn't help",
                (4, 21.3), textcoords="offset points", xytext=(28, 40), fontsize=9, color="#333",
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_xticks(k); ax.set_xlabel("Number of reasoning steps chained together"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 112)
    ax.set_title("How deep can each model reason? A scale-resistant ceiling appears at 4 steps",
                 fontsize=12.5, fontweight="bold", pad=14)
    ax.legend(loc="center right", frameon=False, fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reasoning_depth_wall.png"), dpi=150); plt.close(fig)


# ---------- Figure 3: the research arc (infographic) ----------
def fig_arc():
    steps = [
        ("1. Build", "A 4B-parameter language model\ntrained with NO backpropagation,\non a single GPU"),
        ("2. Measure", "Test what it can learn:\nsentiment ✓  ·  relations ✗\nmulti-step ✗"),
        ("3. Diagnose", "Find the cause: the needed\nstructure is absent from the\nmodel's internal representation"),
        ("4. Control", "Swap only the architecture\n(standard trainable attention +\nbackprop), keep the tasks"),
        ("5. Conclude", "It crosses the limits\n(real SNLI 34%→70%) — the barrier\nwas the design, not the task"),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.4)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 3.4)
    colors = ["#4C72B0", "#4C72B0", "#8172B3", "#DD8452", "#55A868"]
    w, h, gap = 2.25, 2.3, 0.25; x = 0.15
    for (title, body), col in zip(steps, colors):
        ax.add_patch(FancyBboxPatch((x, 0.5), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    fc="white", ec=col, lw=2.2))
        ax.text(x + w/2, 0.5 + h - 0.32, title, ha="center", va="top", fontsize=12, fontweight="bold", color=col)
        ax.text(x + w/2, 0.5 + h - 0.82, body, ha="center", va="top", fontsize=8.6, color="#333")
        if x + w + gap < 12.8:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 0.5 + h/2), (x + w + gap + 0.13, 0.5 + h/2),
                                         arrowstyle="-|>", mutation_scale=16, color="#888", lw=1.6))
        x += w + gap
    ax.set_title("The research in five steps", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "research_arc.png"), dpi=150); plt.close(fig)


# ---------- Figure 4: full scorecard as a heat-style table ----------
def fig_scorecard():
    rows = ["Language modeling", "Sentiment (positive/negative)", "Relational reasoning (SNLI)",
            "Multi-step reasoning", "Grade-school math (GSM8K)"]
    # qualitative capability score 0..1 for the two systems (for shading); text holds the real number
    old_s = [0.62, 0.55, 0.02, 0.02, 0.0]
    new_s = [None, None, 0.70, 0.55, 0.02]
    old_t = ["usable (ppl≈1355)", "79% (with a little BP)", "chance (33%)", "chance (fails 2 steps)", "—"]
    new_t = ["—", "—", "70%", "solves ≤3 steps", "2% (stretch)"]
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.axis("off")
    ax.set_xlim(0, 3); ax.set_ylim(0, len(rows) + 1)
    ax.text(0.02, len(rows) + 0.5, "Capability", fontsize=11, fontweight="bold", va="center")
    ax.text(1.55, len(rows) + 0.5, "Backprop-free 4B model", fontsize=11, fontweight="bold", va="center", ha="center", color=C_OLD)
    ax.text(2.55, len(rows) + 0.5, "Standard model (control)", fontsize=11, fontweight="bold", va="center", ha="center", color=C_NEW)
    import matplotlib.cm as cm
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i + 0.5
        ax.text(0.02, y, r, fontsize=10, va="center")
        for cx, s, t, base in [(1.1, old_s[i], old_t[i], C_OLD), (2.1, new_s[i], new_t[i], C_NEW)]:
            shade = "#f2f2f2" if s is None else cm.RdYlGn(0.15 + 0.7 * s)
            ax.add_patch(plt.Rectangle((cx, y - 0.4), 0.9, 0.8, fc=shade, ec="white"))
            ax.text(cx + 0.45, y, t, ha="center", va="center", fontsize=9,
                    color="#333" if s is not None else "#999")
    ax.set_title("Capability scorecard — what each model design can and cannot do",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "scorecard.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_capability(); fig_depth(); fig_arc(); fig_scorecard()
    print("wrote:", ", ".join(sorted(os.listdir(FIG))))
