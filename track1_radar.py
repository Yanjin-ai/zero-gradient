"""Track 1 -- capability radar: place the ZeroBP 4B stack (and Phase H) on modern-LLM capability axes.

Data-driven from runs/track1_metrics.json (created with defaults if missing) so it regenerates once real
Kaggle eval numbers land -- just edit the JSON and re-run. Each axis is a 0..1 capability score with a
provenance string (real/locked vs synthetic/pending) so the chart never overstates. RESEARCH-ONLY.

  ZeroBP 4B  = locked real numbers (WikiText-103, sentiment 79%, NLI/arith chance).
  Phase H    = G0 synthetic (relational/multi-step = 100%); LM/sentiment marked PENDING (G1/Track1 on GPU).

Run:  python3 track1_radar.py     ->  runs/track1_radar.png
"""
import os, json, math

ROOT = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
METRICS = os.path.join(FIG, "track1_metrics.json"); OUT = os.path.join(FIG, "track1_radar.png")

DEFAULT = {
    "axes": ["Language\nModeling", "Sentiment /\nClassification", "Relational\n(NLI)", "Multi-step\n(Arithmetic)"],
    "series": [
        {"name": "ZeroBP 4B (locked, real)", "values": [0.51, 0.79, 0.334, 0.20],
         "provenance": ["WikiText-103 4B, ~unigram+51%", "4B Mixed-BP embedding 79%",
                        "4B zero-shot = chance 33.4%", "any BP depth = chance 20%"]},
        {"name": "Phase H (real G1/G2, GPU)", "values": [None, None, 0.70, 1.00],
         "provenance": ["not run (Phase H LM pending)", "not run (Phase H sentiment pending)",
                        "REAL SNLI 69.97% (12.4M, T4)", "k<=3 = 100%; k>=4 chance even at 21M (real wall)"]},
    ],
    "note": "Phase H relational = REAL SNLI 69.97% (vs ZeroBP 4B chance 33.4%). Multi-step: installs k<=3 (100%; ZeroBP is chance at ANY depth) but hits a scale-resistant wall at k>=4 (unchanged at 21M/15k). ZeroBP-4B SST-2 = ~chance (BP probe 53%). LM/sentiment axes not yet run for Phase H.",
}


def main():
    os.makedirs(FIG, exist_ok=True)
    if not os.path.exists(METRICS):
        with open(METRICS, "w") as f: json.dump(DEFAULT, f, indent=2)
        print(f"seeded {METRICS} (edit + re-run to refresh with real Kaggle numbers)")
    data = json.load(open(METRICS))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}); metrics JSON is ready at {METRICS} -- render elsewhere."); return

    axes = data["axes"]; N = len(axes)
    ang = [n / N * 2 * math.pi for n in range(N)] + [0.0]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(axes, fontsize=10)
    ax.set_ylim(0, 1); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    for s in data["series"]:
        vals = [v if v is not None else 0.0 for v in s["values"]]; vals += vals[:1]
        ax.plot(ang, vals, linewidth=2, label=s["name"]); ax.fill(ang, vals, alpha=0.12)
    ax.set_title("Capability radar — ZeroBP 4B vs Phase H (research-only)", fontsize=12, pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    fig.text(0.5, 0.02, data.get("note", ""), ha="center", fontsize=7.5, wrap=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")
    for s in data["series"]:
        print(f"  {s['name']}: " + ", ".join(f"{a.strip().splitlines()[0]}={v}" for a, v in zip(axes, s["values"])))


if __name__ == "__main__":
    main()
