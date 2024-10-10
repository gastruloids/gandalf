import shapely
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
import typing as tp
import torch
import json
import skimage as ski
from pathlib import Path
from sklearn.metrics import f1_score, roc_curve, auc
from models import UNet
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass, asdict


def mean(num_list: np.ndarray | tp.List[float]) -> float:
    return float(np.nanmean(num_list))


def _clip(array: np.ndarray):
    """Clip the confidence arrays to binary masks."""
    return (array > 0.2).astype(np.uint8)


def dice_score(gt: np.ndarray, pred: np.ndarray, smooth: float = 1.0) -> float:
    """Dice metric.

    Args:
        gt (np.NDArray): Ground truth
        pred (np.NDArray): Prediction
        smooth (float, optional): Smoothness parameter. Defaults to 1.0.

    Returns:
        float: Dice metric
    """
    # Convert gt to numpy array
    gt = gt.cpu().numpy()
    intersection = np.sum(gt * pred).astype(float)
    return (2.0 * intersection + smooth) / (np.sum(gt) + np.sum(pred) + smooth)


@dataclass
class ClassificationMetrics:
    rod_f1: float
    ring_f1: float
    rod_dice: float
    ring_dice: float
    rod_auc_false_positive: list[float]
    rod_auc_true_positive: list[float]
    ring_auc_false_positive: list[float]
    ring_auc_true_positive: list[float]


@dataclass
class RodsRingsMeasurements:
    # Number of each type
    num_rods: int
    num_rings: int

    # Confidence of each type
    rod_confidence: float
    ring_confidence: float

    # Rod metrics
    rod_perimeters: list[float]
    rod_areas: list[float]

    ring_perimeters: list[float]
    ring_areas: list[float]


def make_polys(contours: tp.List[np.ndarray]) -> tp.List[shapely.Polygon]:
    """Convert a list of contours to a list of shapely polygons."""
    all_polys = []

    for contour in contours:
        poly = shapely.Polygon(contour)

        for other_poly in all_polys:
            if poly.intersects(other_poly):
                # Intersection, merge the two
                new_poly = shapely.symmetric_difference(poly, other_poly)

                # Replace other_poly with new_poly
                all_polys.remove(other_poly)
                all_polys.append(new_poly)
                break

        else:
            # No intersection, add to list
            all_polys.append(poly)

    return all_polys


def get_measurements(
    fuzzy_rods: np.ndarray,
    fuzzy_rings: np.ndarray,
    rods: np.ndarray,
    rings: np.ndarray,
) -> tp.Optional[RodsRingsMeasurements]:
    try:
        rods_confidence = float(np.nanmean(np.where(rods, fuzzy_rods, np.nan)))
        rings_confidence = float(np.nanmean(np.where(rings, fuzzy_rings, np.nan)))

        # Convert rods and rings to polygons
        rod_contours = ski.measure.find_contours(rods, 0.5)
        ring_contours = ski.measure.find_contours(rings, 0.5)

        rod_polys = make_polys(rod_contours)
        ring_polys = make_polys(ring_contours)

        return RodsRingsMeasurements(
            num_rods=len(rod_polys),
            num_rings=len(ring_polys),
            rod_confidence=rods_confidence,
            ring_confidence=rings_confidence,
            rod_perimeters=[poly.length for poly in rod_polys],
            rod_areas=[poly.area for poly in rod_polys],
            ring_perimeters=[poly.length for poly in ring_polys],
            ring_areas=[poly.area for poly in ring_polys],
        )

    except Exception as e:
        warnings.warn(str(e))
        # Return all nan
        return RodsRingsMeasurements(
            num_rods=np.nan,
            num_rings=np.nan,
            rod_confidence=np.nan,
            ring_confidence=np.nan,
            rod_perimeters=[np.nan],
            rod_areas=[np.nan],
            ring_perimeters=[np.nan],
            ring_areas=[np.nan],
        )


def do_transforms(img: Image.Image, crop: bool = True):
    if crop:
        return transforms.Compose(
            [
                transforms.CenterCrop((1024, 1024)),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=128, std=23),
            ]
        )(img)
    else:
        return transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=128, std=23),
            ]
        )(img)


def analyse_image(
    img: Image.Image,
    model_paths: tp.List[Path],
    gt: tp.Tuple[np.ndarray, np.ndarray],
    crop: bool = True,
) -> tp.Tuple[Image.Image, dict]:
    if img.mode == "RGB":
        img = img.convert("L")

    if img.mode == "I;16":
        # Images are 16 bit, convert to 8 bit
        img_array = np.array(img)
        img_array = img_array / 256
        img = Image.fromarray(img_array.astype(np.uint8))

    img = do_transforms(img, crop).unsqueeze(0)

    rings = []
    rods = []

    for model_path in model_paths:
        model = UNet(1, 2)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        rod, ring = model(img)

        rings.append(ring)
        rods.append(rod)

    fuzzy_rings = torch.stack(rings).mean(dim=0).detach().numpy()
    fuzzy_rods = torch.stack(rods).mean(dim=0).detach().numpy()

    rings = _clip(fuzzy_rings)[0]
    rods = _clip(fuzzy_rods)[0]

    measurements = get_measurements(fuzzy_rods, fuzzy_rings, rods, rings)

    fig = plot_data(img, gt, rings, rods, fuzzy_rings, fuzzy_rods)

    if measurements:
        measurements = asdict(measurements)
    else:
        measurements = {}

    return fig, measurements, (fuzzy_rods, fuzzy_rings, rods, rings)


def plot_data(img, gt, rings, rods, fuzzy_rings, fuzzy_rods):
    plt.close("all")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(30, 30))

    for ax, data in zip(
        axes.flatten(),
        (
            gt[0][0],
            gt[1][0],
            None,
            fuzzy_rods[0],
            fuzzy_rings[0],
            None,
            rods,
            rings,
            img[0][0],
        ),
    ):
        ax.set_axis_off()
        if data is None:
            continue
        else:
            ax.imshow(data, cmap="inferno")

    plt.tight_layout()

    return fig


def read_gt_image(gt_path: Path, crop: bool):
    gt_image = np.array(Image.open(gt_path))
    rods = do_transforms(Image.fromarray(gt_image == 1), crop)
    rings = do_transforms(Image.fromarray(gt_image == 2), crop)

    return rods, rings


def make_overall_auc_figure(tprs, fprs):
    """Make an overall AUC figure for the output path.

    Args:
        output_path (Path): The path to the output directory containing model outputs.
    """
    # Make AUC figures
    mean_fpr = np.linspace(0, 1, 100)

    tprs_interp = []
    for fpr, tpr in zip(fprs, tprs):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))

    tprs_interp = np.array(tprs_interp)

    mean_tpr = np.nanmean(tprs_interp, axis=0)  # type: ignore
    std_tpr = np.nanstd(tprs_interp, axis=0)  # type: ignore

    auc_value = auc(mean_fpr, mean_tpr)

    fig = plt.figure()
    plt.plot(mean_fpr, mean_tpr, color="b", label=f"Mean ROC (AUC = {auc_value:.4f})")
    plt.fill_between(
        mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color="grey", alpha=0.2
    )
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    return fig


def timecourse_analysis(folder: Path):
    times = defaultdict(lambda: defaultdict(list))

    for measurements in folder.glob("model_output/measurements/*.json"):
        time = str(measurements.stem).split("_")[1]
        time = parse_time(time)

        with open(measurements) as f:
            f = json.load(f)

        times[time]["num_rings"].append(f["num_rings"])
        times[time]["num_rods"].append(f["num_rods"])
        times[time]["ring_ring_ratio"].append(
            (1 + f["num_rods"]) / (1 + f["num_rings"])
        )
        times[time]["ring_rod_difference"].append(f["num_rings"] - f["num_rods"])
        times[time]["ring_poa"].append(f["average_ring_perimeter_over_area"])
        times[time]["rod_poa"].append(f["average_rod_perimeter_over_area"])

    for key, value in times.items():
        for inner_key, inner_value in value.items():
            times[key][inner_key] = [
                float(np.nanmean(inner_value)),
                float(np.std(inner_value)),
            ]

    with open(folder / "model_output" / "summary_measurements.json", "w") as f:
        json.dump(times, f)


def plot_timecourse_analysis(data_path: Path):
    with open(data_path) as f:
        data = json.load(f)
        means = {k: {j: v[0] for j, v in p.items()} for k, p in data.items()}
        stds = {k: {j: v[1] for j, v in p.items()} for k, p in data.items()}
        means = pd.DataFrame(means)
        stds = pd.DataFrame(stds)

    means = means.T
    stds = stds.T
    means.index = means.index.astype(float)
    stds.index = stds.index.astype(float)
    means = means.sort_index()
    stds = stds.sort_index()
    stds = stds.fillna(0)

    for measurement in means.columns:
        x = means.index
        y = list(means.loc[:, measurement])
        err = list(stds.loc[:, measurement])

        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=err, fmt="o")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(measurement)
        ax.set_title(f"{measurement} over time")
        plt.savefig(data_path.parent / f"{measurement}.png")


def parse_time(time: str):
    if "min" in time:
        return float(time[:-3]) / 60
    elif "hr" in time:
        return float(time[:-2])
    else:
        print("Could not find time")
        return 0


def get_metrics(gt_rods, gt_rings, rods, rings, fuzzy_rods, fuzzy_rings):
    rod_false_positive, rod_true_positive, _ = roc_curve(gt_rods, fuzzy_rods)
    rings_false_positive, rings_true_positive, _ = roc_curve(gt_rings, fuzzy_rings)

    return ClassificationMetrics(
        rod_f1=float(f1_score(gt_rods, rods)),
        ring_f1=float(f1_score(gt_rings, rings)),
        rod_dice=float(dice_score(gt_rods, rods)),
        ring_dice=float(dice_score(gt_rings, rings)),
        rod_auc_false_positive=list(rod_false_positive),
        rod_auc_true_positive=list(rod_true_positive),
        ring_auc_false_positive=list(rings_false_positive),
        ring_auc_true_positive=list(rings_true_positive),
    )


if __name__ == "__main__":
    INFERENCE = False

    timecourse = Path("./data/timecourse/")
    microscope = Path("./data/microscopes/")
    model_paths = list(Path("./output_gamma-280").glob("**/*.pth"))

    for idx, folder in enumerate(
        [
            timecourse,
            microscope,
        ]
    ):
        gt = folder / "gt"
        images = folder / "raw"

        metrics = []

        for image_path in images.glob("*.jpg"):
            gt_path = gt / (image_path.stem + "_annotated.tif")
            gt_rods, gt_rings = read_gt_image(gt_path, crop=False)

            single_image = Image.open(image_path)

            all_measurements = {}

            if INFERENCE:
                fig, measurements, rest = analyse_image(
                    single_image,
                    model_paths=model_paths,
                    gt=(gt_rods, gt_rings),
                    crop=False,
                )

                gt_rods = gt_rods.flatten()
                gt_rings = gt_rings.flatten()

                output_image_path = folder / "model_output" / "images"
                output_image_path.mkdir(exist_ok=True, parents=True)

                fig.savefig(output_image_path / (image_path.stem + ".png"))

                all_measurements[image_path] = measurements

                fuzzy_rods, fuzzy_rings, rods, rings = [a.flatten() for a in rest]

                metr = get_metrics(
                    gt_rods, gt_rings, rods, rings, fuzzy_rods, fuzzy_rings
                )

                with open(
                    folder / "model_output" / "metrics" / (image_path.stem + ".json"),
                    "w",
                ) as f:
                    json.dump(asdict(metr), f)

                metrics.append(metr)

        if INFERENCE:
            with open(folder / "model_output" / "all_metrics.json", "w") as f:
                json.dump([asdict(a) for a in metrics], f)
        else:
            with open(folder / "model_output" / "all_metrics.json") as f:
                metrics = json.load(f)
                metrics = [ClassificationMetrics(**a) for a in metrics]

        summ_rod_dice = np.array([a.rod_dice for a in metrics])
        summ_rod_dice = (np.nanmean(summ_rod_dice), np.std(summ_rod_dice))

        summ_ring_dice = np.array([a.ring_dice for a in metrics])
        summ_ring_dice = (np.nanmean(summ_ring_dice), np.std(summ_ring_dice))

        summ_rod_f1 = np.array([a.rod_f1 for a in metrics])
        summ_rod_f1 = (np.nanmean(summ_rod_f1), np.std(summ_rod_f1))

        summ_ring_f1 = np.array([a.ring_f1 for a in metrics])
        summ_ring_f1 = (np.nanmean(summ_ring_f1), np.std(summ_ring_f1))

        summ_rod_auc = [
            np.interp(
                np.linspace(0, 1, 1000),
                a.rod_auc_false_positive,
                a.rod_auc_true_positive,
            )
            for a in metrics
        ]

        summ_rod_auc = list(np.nanmean(summ_rod_auc, axis=0))

        summ_ring_auc = [
            np.interp(
                np.linspace(0, 1, 1000),
                a.ring_auc_false_positive,
                a.ring_auc_true_positive,
            )
            for a in metrics
        ]

        summ_ring_auc = list(np.nanmean(summ_ring_auc, axis=0))

        out = {
            "ring_dice": summ_ring_dice,
            "rod_dice": summ_rod_dice,
            "rod_f1": summ_rod_f1,
            "ring_f1": summ_ring_f1,
            "rod_auc": summ_rod_auc,
            "ring_auc": summ_ring_auc,
        }

        rod_fig = make_overall_auc_figure(
            tprs=[a.rod_auc_true_positive for a in metrics],
            fprs=[a.rod_auc_false_positive for a in metrics],
        )

        rod_fig.savefig(folder / "model_output" / "rod_auc.png")

        ring_fig = make_overall_auc_figure(
            tprs=[a.ring_auc_true_positive for a in metrics],
            fprs=[a.ring_auc_false_positive for a in metrics],
        )

        ring_fig.savefig(folder / "model_output" / "ring_auc.png")

        with open(folder / "model_output" / "master_metric.json", "w") as f:
            json.dump(out, f)

            if "timecourse" in str(folder):
                timecourse_analysis(folder)
                plot_timecourse_analysis(
                    folder / "model_output" / "summary_measurements.json"
                )
