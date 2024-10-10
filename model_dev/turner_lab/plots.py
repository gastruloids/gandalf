from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
import json
from scipy import stats

from turner_lab.test import get_measurements, RodsRingsMeasurements, do_transforms


def _clip(array: np.ndarray):
    """Clip the confidence arrays to binary masks."""
    return (array > 0.2).astype(np.uint8)


def training_progress_plot(training_progress_dir: Path):
    # Load the training progress data
    fig, ax = plt.subplots(1, 1)
    for i in range(5):
        for dataset in ["training", "validation"]:
            losses = pd.read_csv(
                training_progress_dir / f"fold_{i}_log_{dataset}.csv"
            ).loc[:, "Value"]

            ls = "-" if dataset == "training" else "--"
            ax.plot(
                losses, label=f"fold_{i}_{dataset}", alpha=0.5, color=f"C{i}", ls=ls
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    plt.tight_layout()

    fig.savefig(training_progress_dir / "training_progress.png")


def training_metrics(training_dir: Path):
    fig, ax = plt.subplots(1, 1)

    with open(training_dir / "master_metrics.json") as f:
        master_metrics = json.load(f)

    # Plot the metrics
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Dice Score", "F1 Score"])
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")

    w = 0.3
    bars = ax.bar(
        [0 - w / 2, 0 + w / 2, 1 - w / 2, 1 + w / 2],
        [
            master_metrics["ring_dice"][0],
            master_metrics["rod_dice"][0],
            master_metrics["ring_f1"][0],
            master_metrics["rod_f1"][0],
        ],
        yerr=[
            master_metrics["ring_dice"][1],
            master_metrics["rod_dice"][1],
            master_metrics["ring_f1"][1],
            master_metrics["rod_f1"][1],
        ],
        width=w,
        color=["C0", "C1", "C0", "C1"],
    )

    for idx, metric in enumerate(["ring_dice", "rod_dice", "ring_f1", "rod_f1"]):
        bar = bars[idx]

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.1,
            f"{master_metrics[metric][0]:.4f} +/-  \n {master_metrics[metric][1]:.4f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    legend_dict = {
        "Rod": "C1",
        "Ring": "C0",
    }
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, fc=color) for color in legend_dict.values()],
        legend_dict.keys(),
    )

    fig.savefig(training_dir / "training_metrics.png")


def get_r2_plots(annotation_dir: Path, model_dir: Path):
    compare_measurements = {
        "gt_num_rods": [],
        "gt_num_rings": [],
        "model_num_rods": [],
        "model_num_rings": [],
        "gt_avg_rod_perimeter": [],
        "gt_avg_ring_perimeter": [],
        "model_avg_rod_perimeter": [],
        "model_avg_ring_perimeter": [],
        "gt_avg_rod_area": [],
        "gt_avg_ring_area": [],
        "model_avg_rod_area": [],
        "model_avg_ring_area": [],
        "gt_ring_rod_ratio": [],
        "model_ring_rod_ratio": [],
        "gt_avg_rod_poa": [],
        "model_avg_rod_poa": [],
        "gt_avg_ring_poa": [],
        "model_avg_ring_poa": [],
    }
    for gt_annotation in annotation_dir.glob("*.tif"):
        gt_annotation_stem = gt_annotation.stem[: -len("_annotated")]

        # Calculate the measurements for the annotation
        gt_image = np.array(Image.open(gt_annotation))
        cropped = "microscope" in str(model_dir)
        rods = do_transforms(Image.fromarray(gt_image == 1), cropped)[0].numpy()
        rings = do_transforms(Image.fromarray(gt_image == 2), cropped)[0].numpy()

        # Find the measurements file for the model output
        gt_measurements: RodsRingsMeasurements = get_measurements(
            rods, rings, rods, rings
        )

        compare_measurements["gt_num_rods"].append(gt_measurements.num_rods)
        compare_measurements["gt_num_rings"].append(gt_measurements.num_rings)
        compare_measurements["gt_avg_rod_perimeter"].append(
            np.nanmean(gt_measurements.rod_perimeters)
        )
        compare_measurements["gt_avg_ring_perimeter"].append(
            np.nanmean(gt_measurements.ring_perimeters)
        )
        compare_measurements["gt_avg_rod_area"].append(
            np.nanmean(gt_measurements.rod_areas)
        )
        compare_measurements["gt_avg_ring_area"].append(
            np.nanmean(gt_measurements.ring_areas)
        )
        compare_measurements["gt_ring_rod_ratio"].append(
            (1 + gt_measurements.num_rods) / (1 + gt_measurements.num_rings)
        )
        compare_measurements["gt_avg_rod_poa"].append(
            np.nanmean(gt_measurements.rod_perimeters)
            / np.nanmean(gt_measurements.rod_areas)
        )
        compare_measurements["gt_avg_ring_poa"].append(
            np.nanmean(gt_measurements.ring_perimeters)
            / np.nanmean(gt_measurements.ring_areas)
        )

        if "annotated" in str(annotation_dir):
            # We need to calculate the measurements from the model file.
            print(gt_annotation_stem)
            model_ring_mask_path = list(
                model_dir.glob(f"**/{gt_annotation_stem}.tif_ring.npy")
            )[0]
            model_rods_mask_path = list(
                model_dir.glob(f"**/{gt_annotation_stem}.tif_rod.npy")
            )[0]

            # Load numpy file
            fuzzy_rods = np.load(model_rods_mask_path)
            fuzzy_rings = np.load(model_ring_mask_path)
            rods, rings = _clip(fuzzy_rods), _clip(fuzzy_rings)

            # Calculate measurements
            model_measurements: RodsRingsMeasurements = get_measurements(
                fuzzy_rods, fuzzy_rings, rods, rings
            )

            compare_measurements["model_num_rods"].append(model_measurements.num_rods)
            compare_measurements["model_num_rings"].append(model_measurements.num_rings)
            compare_measurements["model_avg_rod_perimeter"].append(
                np.nanmean(model_measurements.rod_perimeters)
            )
            compare_measurements["model_avg_ring_perimeter"].append(
                np.nanmean(model_measurements.ring_perimeters)
            )
            compare_measurements["model_avg_rod_area"].append(
                np.nanmean(model_measurements.rod_areas)
            )
            compare_measurements["model_avg_ring_area"].append(
                np.nanmean(model_measurements.ring_areas)
            )
            compare_measurements["model_ring_rod_ratio"].append(
                (1 + model_measurements.num_rods) / (1 + model_measurements.num_rings)
            )
            compare_measurements["model_avg_rod_poa"].append(
                np.nanmean(model_measurements.rod_perimeters)
                / np.nanmean(model_measurements.rod_areas)
            )
            compare_measurements["model_avg_ring_poa"].append(
                np.nanmean(model_measurements.ring_perimeters)
                / np.nanmean(model_measurements.ring_areas)
            )

        else:
            # We can load the measurements directly
            model_measurements_path = model_dir / (gt_annotation_stem + ".json")
            with open(model_measurements_path) as f:
                model_measurements = json.load(f)

            compare_measurements["model_num_rods"].append(
                model_measurements["num_rods"]
            )
            compare_measurements["model_num_rings"].append(
                model_measurements["num_rings"]
            )
            compare_measurements["model_avg_rod_perimeter"].append(
                np.nanmean(model_measurements["rod_perimeters"])
            )
            compare_measurements["model_avg_ring_perimeter"].append(
                np.nanmean(model_measurements["ring_perimeters"])
            )
            compare_measurements["model_avg_rod_area"].append(
                np.nanmean(model_measurements["rod_areas"])
            )
            compare_measurements["model_avg_ring_area"].append(
                np.nanmean(model_measurements["ring_areas"])
            )
            compare_measurements["model_ring_rod_ratio"].append(
                (1 + model_measurements["num_rods"])
                / (1 + model_measurements["num_rings"])
            )
            compare_measurements["model_avg_rod_poa"].append(
                np.nanmean(model_measurements["rod_perimeters"])
                / np.nanmean(model_measurements["rod_areas"])
            )
            compare_measurements["model_avg_ring_poa"].append(
                np.nanmean(model_measurements["ring_perimeters"])
                / np.nanmean(model_measurements["ring_areas"])
            )

    for measurement in [
        "num_rods",
        "num_rings",
        "avg_rod_perimeter",
        "avg_ring_perimeter",
        "avg_rod_area",
        "avg_ring_area",
        "ring_rod_ratio",
        "avg_rod_poa",
        "avg_ring_poa",
    ]:
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        ax.scatter(
            compare_measurements[f"gt_{measurement}"],
            compare_measurements[f"model_{measurement}"],
        )

        # Calculate R2
        gt = np.array(compare_measurements[f"gt_{measurement}"])
        model = np.array(compare_measurements[f"model_{measurement}"])

        # Remove Nans values from both arrays
        gt_filtered = gt[~(np.isnan(gt) | np.isnan(model))]
        model_filtered = model[~(np.isnan(gt) | np.isnan(model))]

        ax.set_xlabel(f"Ground Truth {' '.join(measurement.split('_')).capitalize()}")
        ax.set_ylabel(f"Model {' '.join(measurement.split('_')).capitalize()}")
        plt.tight_layout()

        # Calculate R2
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            gt_filtered, model_filtered
        )

        ax.plot(
            [np.min(gt_filtered), np.max(gt_filtered)],
            [
                slope * np.min(gt_filtered) + intercept,
                slope * np.max(gt_filtered) + intercept,
            ],
            color="black",
        )

        ax.text(
            0.05,
            0.95,
            f"R2: {r_value**2:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(model_dir / f"{measurement}_r2.png")


if __name__ == "__main__":
    # R2 Scores
    get_r2_plots(
        Path("data/microscopes/gt/"),
        Path("data/microscopes/model_output_croped/measurements"),
    )

    get_r2_plots(
        Path("data/timecourse/gt/"), Path("data/timecourse/model_output/measurements")
    )

    get_r2_plots(Path("data/annotated/"), Path("output_gamma-280/"))

    # Training Progress
    training_progress_plot(Path("output_gamma-280/training_progress"))

    # Metrics
    training_metrics(Path("output_gamma-280"))
    training_metrics(Path("data/microscopes/model_output_croped/"))
    training_metrics(Path("data/microscopes/model_output/"))
    training_metrics(Path("data/timecourse/model_output/"))
