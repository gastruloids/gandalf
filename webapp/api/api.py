import base64
import json
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import skimage
import shapely
import matplotlib.pyplot as plt
import numpy as np
import typing as tp
import numpy.typing as npt
import torch
from shapely.geometry import Polygon
from flask import Flask, request, send_file
from models import UNet
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass, asdict

app = Flask(__name__)

model_paths = list(Path("./models").rglob("*.pth"))


def mean(num_list: npt.NDArray | tp.List[float]) -> float:
    return float(np.nanmean(num_list))


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
    rod_perimeter_over_area: list[float]

    average_rod_perimeter: float
    average_rod_area: float
    average_rod_perimeter_over_area: float

    ring_perimeters: list[float]
    ring_areas: list[float]
    ring_perimeter_over_area: list[float]

    average_ring_perimeter: float
    average_ring_area: float
    average_ring_perimeter_over_area: float
    average_ring_no_holes: float


def convert_from_base_64(string: str):
    data = string.split(",")[1]
    return base64.b64decode(data)


def _clip(array: npt.NDArray):
    """Clip the confidence arrays to binary masks."""
    return (array > 0.5).astype(np.uint8)


def get_measurements(
    fuzzy_rods: npt.NDArray,
    fuzzy_rings: npt.NDArray,
    rods: npt.NDArray,
    rings: npt.NDArray,
) -> tp.Optional[RodsRingsMeasurements]:
    try:
        rods_confidence = float(np.nanmean(np.where(rods, fuzzy_rods, np.nan)))
        rings_confidence = float(np.nanmean(np.where(rings, fuzzy_rings, np.nan)))

        # Convert rods and rings to polygons
        rod_contours = skimage.measure.find_contours(rods, 0.5)
        ring_contours = skimage.measure.find_contours(rings, 0.5)

        rod_polys = make_polys(rod_contours)
        ring_polys = make_polys(ring_contours)

        num_rods = len(rod_polys)
        num_rings = len(ring_polys)

        rod_perims = [poly.length for poly in rod_polys]
        rod_areas = [poly.area for poly in rod_polys]
        rod_perimeter_over_area = [poly.length / poly.area for poly in rod_polys]

        avg_rod_perimeter = mean(rod_perims)
        avg_rod_area = mean(rod_areas)
        avg_rod_perimeter_over_area = mean(rod_perimeter_over_area)

        ring_perims = [poly.length for poly in ring_polys]
        ring_areas = [poly.area for poly in ring_polys]
        ring_perimeter_over_area = [poly.length / poly.area for poly in ring_polys]

        avg_ring_perimeter = mean(ring_perims)
        avg_ring_area = mean(ring_areas)
        avg_ring_perfimeter_over_area = mean(ring_perimeter_over_area)

        average_ring_no_holes = mean([len(poly.interiors) for poly in ring_polys])

        return RodsRingsMeasurements(
            num_rods=num_rods,
            num_rings=num_rings,
            rod_confidence=rods_confidence,
            ring_confidence=rings_confidence,
            rod_perimeters=rod_perims,
            rod_areas=rod_areas,
            rod_perimeter_over_area=rod_perimeter_over_area,
            average_rod_perimeter=avg_rod_perimeter,
            average_rod_area=avg_rod_area,
            average_rod_perimeter_over_area=avg_rod_perimeter_over_area,
            ring_perimeters=ring_perims,
            ring_areas=ring_areas,
            ring_perimeter_over_area=ring_perimeter_over_area,
            average_ring_perimeter=avg_ring_perimeter,
            average_ring_area=avg_ring_area,
            average_ring_perimeter_over_area=avg_ring_perfimeter_over_area,
            average_ring_no_holes=average_ring_no_holes,
        )

    except Exception as e:
        print(e)
        return None


def make_polys(contours: tp.List[npt.NDArray]) -> tp.List[Polygon]:
    """Convert a list of contours to a list of shapely polygons."""
    all_polys = []

    for contour in contours:
        poly = Polygon(contour)

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


def analyse_image(
    img: Image.Image,
) -> tp.Tuple[BytesIO, dict]:
    if img.mode == "RGB":
        img = img.convert("L")

    if img.mode == "I;16":
        # Images are 16 bit, convert to 8 bit
        img_array = np.array(img)
        img_array = img_array / 256
        img = Image.fromarray(img_array.astype(np.uint8))

    img = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])(
        img
    )

    img = img.unsqueeze(0)

    rings = []
    rods = []

    for model_path in list(model_paths):
        model = UNet(1, 2)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        rod, ring = model(img)

        rings.append(ring)
        rods.append(rod)

    fuzzy_rings = torch.stack(rings).mean(dim=0).detach().numpy()
    fuzzy_rods = torch.stack(rods).mean(dim=0).detach().numpy()

    rings = _clip(fuzzy_rings)
    rods = _clip(fuzzy_rods)

    all = rings * 0.5 + rods

    measurements = get_measurements(fuzzy_rods, fuzzy_rings, rods[0], rings[0])

    plt.close("all")

    fig, ax = plt.subplots(figsize=(10, 10))

    all = all[0]

    ax.imshow(all, cmap="inferno")

    # Hide axis
    ax.set_axis_off()

    plt.tight_layout()

    buffer = BytesIO()

    plt.savefig(buffer, format="png")

    if measurements:
        measurements = asdict(measurements)
    else:
        measurements = {}

    return buffer, measurements


def analyse_files_in_zip(zipdata: BytesIO) -> tp.Dict[str, str]:
    files = {}
    with zipfile.ZipFile(zipdata, "r") as zip_ref:
        for file in zip_ref.namelist():
            with zip_ref.open(file) as f:
                img = Image.open(f)
                img, measurements = analyse_image(img)

                measurements = json.dumps(measurements, indent=4)

                files[file] = (img.getvalue(), measurements)

    return files


def build_new_zip(files: tp.Dict[str, tp.Tuple[str, str]]) -> BytesIO:
    out_zip = BytesIO()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, (img, measurements) in files.items():
            # Make a buffered image file
            filename = ".".join(filename.split(".")[:-1])
            data = zipfile.ZipInfo(
                f"{filename}_img.png", date_time=datetime.now().timetuple()[:6]
            )
            zf.writestr(data, img)

            data = zipfile.ZipInfo(
                f"{filename}_measurements.json",
                date_time=datetime.now().timetuple()[:6],
            )
            zf.writestr(data, measurements)

    out_zip.seek(0)

    return out_zip


@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    # Load JSON data
    decoded = convert_from_base_64(request.form["zip"])

    # First, parse the zip file.
    files = analyse_files_in_zip(BytesIO(decoded))

    out_zip = build_new_zip(files)

    return send_file(out_zip, as_attachment=True, download_name="analysis.zip")


@app.route("/api/predict", methods=["POST"])
def predict():
    # Convert image from base64 to numpy array
    decoded = convert_from_base_64(request.form["image"])
    image = Image.open(BytesIO(decoded))

    img, measurements = analyse_image(image)

    img_str = base64.encodebytes(img.getvalue()).decode("utf-8")

    # Return prediction
    return {"image": img_str, "measurements": measurements}


if __name__ == "__main__":
    app.run(port=5000)
