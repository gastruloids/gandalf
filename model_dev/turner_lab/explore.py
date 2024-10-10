"""Module for exploring the data."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from shapely.geometry import Polygon
import json
import typing as tp
import numpy.typing as npt


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


def parse_file(filename: Path) -> tp.List[Annotation]:
    """Parse a single file and ouput a list of the shapes within it.

    Args:
        filename (Path): File to parse

    Returns:
        tp.List[Annotation]: List of annotations in the file
    """
    with open(filename, "r") as f:
        data = json.load(f)

    polygons = []

    for annotation in data["annotations"]:
        w = annotation["bounding_box"]["w"]
        h = annotation["bounding_box"]["h"]
        label = annotation["name"]

        # Make the polygon of the points
        points = parse_points(annotation["polygon"]["paths"])

        polygon = Annotation(
            w=w,
            h=h,
            label=label,
            polygon=points,
        )

        polygons.append(polygon)

    return polygons


def parse_points(polygon: tp.List) -> Polygon:
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
        return Polygon(largest.exterior, [p.exterior for p in paths if p != largest])


def run():
    """Run loop for all files."""
    all_polygons = []

    for filename in Path("./data/annotations/gt").glob("*.json"):
        polygons = parse_file(filename)
        all_polygons.extend(polygons)

    all_features = np.array(
        [list(polygon.get_metrics().values()) for polygon in all_polygons]
    )

    # Make a T-SNE plot of the point in space
    make_tsne_plot(all_features, all_polygons)
    labels = [1 if p.label == "ring" else 0 for p in all_polygons]

    # Also look at a Decision Tree
    clf = DecisionTreeClassifier()
    clf.fit(all_features, labels)

    plot_tree(clf, filled=True, feature_names=list(polygons[0].get_metrics().keys()))
    plt.savefig("test_tree.png")


def make_tsne_plot(all_features: npt.ArrayLike, polygons: tp.List[Annotation]):
    """Make a T-SNE plot with SVM classifier to loop at class separation.

    Args:
        all_features (npt.ArrayLike): Array of metrics to use as features for T-SNE
        polygons (tp.List[Annotation]): Annotation objects (for labels)
    """
    transformed_features = StandardScaler().fit_transform(all_features)

    pca = TSNE(n_components=2)
    transformed_features = pca.fit_transform(transformed_features)

    fig, ax = plt.subplots()

    ax.scatter(
        transformed_features[:, 0],
        transformed_features[:, 1],
        c=["red" if p.label == "ring" else "blue" for p in polygons],
    )

    # SVM decision boundary
    clf = SVC()
    clf.fit(transformed_features, [1 if p.label == "ring" else 0 for p in polygons])
    x_max, x_min = (
        transformed_features[:, 0].max() + 1,
        transformed_features[:, 0].min() - 1,
    )
    y_max, y_min = (
        transformed_features[:, 1].max() + 1,
        transformed_features[:, 1].min() - 1,
    )

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    fig.savefig("test_pca.png")

    # Predict all points, for those that are wrong, plot them
    wrong_points = []
    labels = [1 if p.label == "ring" else 0 for p in polygons]

    for i, point in enumerate(transformed_features):
        if clf.predict([point]) != labels[i]:
            wrong_points.append(i)

    plot_polygons([polygons[i] for i in wrong_points])


def plot_polygons(polygons: tp.List[Annotation]):
    """Plot a list of annotations to quickly look at their shape.

    Args:
        polygons (tp.List[Annotation]): A list of annotations to show
    """
    _, ax = plt.subplots()

    for polygon in polygons:
        c = "red" if polygon.label == "ring" else "blue"
        x, y = polygon.polygon.exterior.xy
        ax.plot(x, y, label=polygon.label, c=c)

    plt.savefig("test.png")


if __name__ == "__main__":
    run()
