"""Tools for analyzing model outputs."""

from pathlib import Path
import numpy as np
import typing as tp
import numpy.typing as npt
from PIL import Image
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from sklearn.metrics import roc_curve, auc
import json


def make_metric_figures(output_path: Path):
    """Make metric figures for the output path.

    Args:
        output_path (Path): The path to the output directory containing model outputs.
    """
    folds = output_path.glob("fold*")

    master_metrics: tp.Dict[str, tp.List[float]] = {
        "ring_f1": [],
        "rod_f1": [],
        "ring_dice": [],
        "rod_dice": [],
    }

    for fold in folds:
        metric_path = fold / "metrics.json"

        with open(metric_path, "r") as f:
            metrics = json.load(f)

            # Yuck!
            master_metrics["ring_f1"].append(metrics["ring_f1"])
            master_metrics["rod_f1"].append(metrics["rod_f1"])
            master_metrics["ring_dice"].append(metrics["ring_dice"])
            master_metrics["rod_dice"].append(metrics["rod_dice"])

    summary_metrics = {
        "ring_f1": (
            np.mean(master_metrics["ring_f1"]),
            np.std(master_metrics["ring_f1"]),
        ),
        "rod_f1": (np.mean(master_metrics["rod_f1"]), np.std(master_metrics["rod_f1"])),
        "ring_dice": (
            np.mean(master_metrics["ring_dice"]),
            np.std(master_metrics["ring_dice"]),
        ),
        "rod_dice": (
            np.mean(master_metrics["rod_dice"]),
            np.std(master_metrics["rod_dice"]),
        ),
    }

    with open(output_path / "master_metrics.json", "w") as f:
        json.dump(summary_metrics, f)


def make_overall_auc_figure(output_path: Path):
    """Make an overall AUC figure for the output path.

    Args:
        output_path (Path): The path to the output directory containing model outputs.
    """
    rod_fprs, rod_tprs = get_auc_values(output_path, "rod")
    ring_fprs, ring_tprs = get_auc_values(output_path, "ring")

    # Make AUC figures
    for fprs, tprs, shape in zip(
        [rod_fprs, ring_fprs], [rod_tprs, ring_tprs], ["rod", "ring"]
    ):
        mean_fpr = np.linspace(0, 1, 100)

        tprs_interp = []
        for fpr, tpr in zip(fprs, tprs):
            tprs_interp.append(np.interp(mean_fpr, fpr, tpr))

        tprs_interp = np.array(tprs_interp)

        mean_tpr = tprs_interp.mean(axis=0)  # type: ignore
        std_tpr = tprs_interp.std(axis=0)  # type: ignore

        auc_value = auc(mean_fpr, mean_tpr)

        plt.figure()
        plt.plot(
            mean_fpr, mean_tpr, color="b", label=f"Mean ROC (AUC = {auc_value:.4f})"
        )
        plt.fill_between(
            mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color="grey", alpha=0.2
        )
        plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{shape.capitalize()} AUC")
        plt.legend()
        plt.savefig(output_path / f"{shape}_auc.png")
        plt.close()


def get_auc_values(
    output_path: Path, shape: str
) -> tp.Tuple[tp.List[float], tp.List[float]]:
    """Get the AUC values for a given shape.

    Args:
        output_path (Path): The path to the output directory containing model outputs.
        shape (str): The shape to get AUC values for.

    Returns:
        Tuple[List[float], List[float]]: The FPRs and TPRs for the given shape.
    """
    dataset = Path("./data/raw")
    fprs, tprs = [], []

    for raw_image in dataset.glob("*"):
        name = raw_image.name

        mask = Path(f"./data/masks/{shape}s/{raw_image.stem}.png")
        mask = (np.array(Image.open(mask).resize((512, 512))) > 128).astype(int)

        fuzzy_paths = list(output_path.glob(f"**/{name}_{shape}.npy"))

        if len(fuzzy_paths) == 0:
            print(f"No fuzzy path found for {name}")
            continue

        fuzzy = np.load(fuzzy_paths[0])

        fpr, tpr, _ = roc_curve(mask.flatten(), fuzzy.flatten())  # type: ignore

        fprs.append(fpr)
        tprs.append(tpr)

    return fprs, tprs


if __name__ == "__main__":
    output_path = Path("./output_gamma-280/")
    make_metric_figures(output_path)
    make_overall_auc_figure(output_path)
