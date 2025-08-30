import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def reliability_plot(y_true, y_prob, outpath: Path, n_bins=15, title="Reliability (Calibration)"):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    px, py = [], []
    for i in range(n_bins):
        mask = binids == i
        if mask.any():
            px.append((bins[i] + bins[i+1]) / 2)
            py.append(y_true[mask].mean())
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(px, py, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
