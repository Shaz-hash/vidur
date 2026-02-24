#!/usr/bin/env python3
import matplotlib.pyplot as plt

# Policy label -> SLO cost
data = {
    "SJF, 512": 5.628,
    "SJF, 1024": 9.642,
    "LST, 512": 9.531,
    "LST, 1024": 10.617,
    "Model_MCTS_Policy": 5.539,
}

title = "MCTS prior using model after training with 16k samples"
out_path = "policy_slo_cost_bar.png"

labels = list(data.keys())
values = list(data.values())

colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(labels, values, color=colors, edgecolor="black")

ax.set_ylabel("SLO Cost")
ax.set_xlabel("Policy")
ax.set_title(title)
ax.grid(axis="y", linestyle="--", alpha=0.35)

# Label each bar with its SLO cost
for bar, v in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v + 0.08,
        f"{v:.5f}".rstrip("0").rstrip("."),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(out_path, dpi=180)
print(f"Saved: {out_path}")
