# diagram_dtd.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def add_node(ax, xy, text, color="skyblue"):
    box = dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.8, edgecolor="black")
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=9,
            fontweight="bold", bbox=box, zorder=2)

def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", 
                            mutation_scale=15, linewidth=1.5, color="black", zorder=1)
    ax.add_patch(arrow)

fig, ax = plt.subplots(figsize=(8, 10))
ax.axis("off")

# Fixer les limites visibles de la figure
ax.set_xlim(-2, 7)
ax.set_ylim(0, 9)

# Positions des blocs
pos = {
    "images": (0, 8),
    "labels": (5, 8),
    "patches": (2.5, 6.5),
    "jpeg": (2.5, 5.5),
    "model": (2.5, 4.5),
    "seg": (2.5, 3.5),
    "loss": (0, 2.5),
    "optim": (2.5, 1.5),
    "val": (5, 3),
    "ckpt": (5, 1.5)
}

# Ajout des nœuds
add_node(ax, pos["images"], "Images (.jpg)", "lightblue")
add_node(ax, pos["labels"], "Labels (.png)", "lightgreen")
add_node(ax, pos["patches"], "Découpage\nsous-patchs (512x512)", "lightblue")
add_node(ax, pos["jpeg"], "JPEG features\n(DCT + Q-tables)", "orange")
add_node(ax, pos["model"], "seg_dtd\n(Swin gelé + heads)", "violet")
add_node(ax, pos["seg"], "Segmentation binaire\n(0=authentique, 1=falsifié)", "pink")
add_node(ax, pos["val"], "Validation\n(IoU, Precision, Recall, F1)", "lightyellow")

# Ajout des flèches
add_arrow(ax, pos["images"], pos["patches"])
add_arrow(ax, pos["labels"], pos["patches"])
add_arrow(ax, pos["patches"], pos["jpeg"])
add_arrow(ax, pos["jpeg"], pos["model"])
add_arrow(ax, pos["model"], pos["seg"])
add_arrow(ax, pos["model"], pos["val"])

plt.tight_layout()
plt.savefig("diagram_dtd_pipeline.png", dpi=300)
plt.show()
