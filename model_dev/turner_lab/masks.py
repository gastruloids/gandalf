"""Support for creating masks from annotations."""

import typing as tp
import numpy as np
from shapely import Polygon
from PIL import Image, ImageDraw, ImageChops
import json
from pathlib import Path
import warnings


class Annotation:
    """Class for repackaging each annotation into important features."""

    def __init__(self, w: float, h: float, label: str, polygon: Polygon):
        """Initialise the Annotation object.

        Args:
            w (float): Width of the annotation
            h (float): Height of the annotation
            label (str): Ring/Rod label
            polygon (Polygon): Polygon object of all the points in the annotation
        """
        self.w = w
        self.h = h
        self.label = label
        self.polygon = polygon

    def get_metrics(self) -> tp.Dict[str, float]:
        """Get some summary metrics of the annotation.

        Returns:
            tp.Dict[str, float]: Metrics along with their values
        """
        return {
            "w": self.w,
            "h": self.h,
            "area": self.polygon.area,
            "perimeter": self.polygon.length,
            "wh_ratio": self.w / self.h,
            "ap_ratio": self.polygon.area / self.polygon.length,
            "holes": len(self.polygon.interiors),
        }


class TifMask:
    """Class for handling the mask from the new TIF format."""

    def __init__(self, image: Image.Image, annotation_image: Image.Image):
        """Initialise the Mask object.

        Args:
            image (Image.Image): Image of the original image.
            annotation_image (Image.Image): Image of the annotations
        """
        self.image = image
        self.annotation_image = annotation_image

    @classmethod
    def from_files(cls, img_path: Path, mask_path: Path) -> "TifMask":
        """Create a mask from the file.

        Args:
            img_path (Path): Path to the image file
            mask_path (Path): Path to the mask file

        Returns:
            Mask: New mask object
        """
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        return cls(img, mask)

    def to_file(self, output_path: Path):
        """Save the mask to a file."""
        mask_array = np.array(self.annotation_image)

        rods = (mask_array == 1).astype(np.uint8) * 255
        rings = (mask_array == 2).astype(np.uint8) * 255

        rod_mask = Image.fromarray(rods)
        ring_mask = Image.fromarray(rings)

        both_mask = ImageChops.lighter(rod_mask, ring_mask)

        original_path = Path(self.image.filename).name

        rod_mask.save((output_path / "rods" / original_path).with_suffix(".png"))
        ring_mask.save((output_path / "rings" / original_path).with_suffix(".png"))
        both_mask.save((output_path / "both" / original_path).with_suffix(".png"))


class JsonMask:
    """Class for handling the mask and image together."""

    def __init__(self, image: Image.Image, annotation: tp.List[Annotation]):
        """Initialise the Mask object.

        Args:
            image (Image.Image): Image of the original image.
            annotation (tp.List[Annotation]): List of annotations in the image
        """
        self.image = image
        self.annotation = annotation

    @classmethod
    def from_file(cls, img_path: Path, mask_path: Path) -> "JsonMask":
        """Create a mask from the file.

        Args:
            img_path (Path): Path to the image file
            mask_path (Path): Path to the mask file

        Returns:
            Mask: New mask object
        """
        img = Image.open(img_path)
        mask = cls._parse_json_mark(mask_path)

        return cls(img, mask)

    def to_file(self, output_path: Path):
        """Save the mask to a file.

        Args:
            output_path (Path): Path to save the mask to
        """
        # Save the mask
        rod_mask, ring_mask, both_mask = self.make_image_mask()

        original_path = Path(self.image.filename).name

        rod_mask.save((output_path / "rods" / original_path).with_suffix(".png"))

        ring_mask.save((output_path / "rings" / original_path).with_suffix(".png"))

        both_mask.save((output_path / "both" / original_path).with_suffix(".png"))

    def make_image_mask(self) -> Image.Image:
        """Create a mask image from the annotations.

        Returns:
            Image.Image: Image of the mask
        """
        rod_mask = Image.new("L", self.image.size, 0)
        ring_mask = Image.new("L", self.image.size, 0)

        for annotation in self.annotation:
            mask = rod_mask if annotation.label == "rod" else ring_mask

            # Draw exterior polygon
            ImageDraw.Draw(mask).polygon(
                annotation.polygon.exterior.coords, outline=255, fill=255
            )

            # Add holes
            for hole in annotation.polygon.interiors:
                ImageDraw.Draw(mask).polygon(hole.coords, outline=0, fill=0)

        both_mask = ImageChops.lighter(rod_mask, ring_mask)

        return rod_mask, ring_mask, both_mask

    @staticmethod
    def _parse_json_mark(mask_path: Path) -> tp.List[Annotation]:
        """Parse a single file and ouput a list of the shapes within it.

        Args:
            filename (Path): File to parse

        Returns:
            tp.List[Annotation]: List of annotations in the file
        """
        with open(mask_path, "r") as f:
            data = json.load(f)

        polygons = []

        for annotation in data["annotations"]:
            w = annotation["bounding_box"]["w"]
            h = annotation["bounding_box"]["h"]
            label = annotation["name"]

            # Make the polygon of the points
            points = JsonMask._parse_points(annotation["polygon"]["paths"])

            polygon = Annotation(
                w=w,
                h=h,
                label=label,
                polygon=points,
            )

            polygons.append(polygon)

        return polygons

    @staticmethod
    def _parse_points(polygon: tp.List) -> Polygon:
        """Parse an annotation points into a polygon.

        Args:
            polygon (tp.List): List of paths of points

        Returns:
            Polygon: Shapely Polygon object describing the annotation
        """
        if len(polygon) == 1:
            # Only one path, make the polygon
            points = [(point["x"], point["y"]) for point in polygon[0]]
            return Polygon(points)

        else:
            # Multiple paths, make numerous paths
            paths = []
            for path in polygon:
                points = [(point["x"], point["y"]) for point in path]
                paths.append(Polygon(points))

            # Find the largest polygon, this is probably the outside
            largest = max(paths, key=lambda p: p.area)

            # Make new polygon with holes
            return Polygon(
                largest.exterior, [p.exterior for p in paths if p != largest]
            )


if __name__ == "__main__":

    # Old JSON method
    # for img_path in Path("./data/images").glob("*.jpg"):
    #     print(img_path)
    #     mask_path = Path("./data/annotations/gt") / (img_path.stem + ".json")

    #     mask = JsonMask.from_file(img_path, mask_path)
    #     mask.to_file(Path("./data/masks"))

    # New TIF method
    for img_path in Path("./data/raw").glob("*.tif"):
        mask_path = Path("./data/annotated") / f"{img_path.stem}_annotated.tif"

        if not mask_path.exists():
            warnings.warn(f"Mask {mask_path} not found for {img_path}")
            continue

        mask = TifMask.from_files(img_path, mask_path)
        mask.to_file(Path("./data/masks"))
