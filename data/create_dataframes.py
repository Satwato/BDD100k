import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import cv2  # OpenCV for image operations
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm  # For progress bars in notebooks

# Make plots look nicer
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

IMG_WIDTH, IMG_HEIGHT = 1280, 720
IMG_AREA = IMG_WIDTH * IMG_HEIGHT
TIGHTNESS_THRESHOLD = 0.5


def flatten_keys(d, parent_key=""):
    keys = []
    for k, v in d.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            keys.extend(flatten_keys(v, full_key))
        else:
            keys.append(full_key)
    return keys


class BddParser:
    """
    Parses BDD100k data, creating a rich feature set for deep analysis.
    """

    def __init__(self, json_path: Path):
        print(f"Loading data from {json_path}...")
        with open(json_path, "r") as f:
            self.raw_data: List[Dict[str, Any]] = json.load(f)
        print(f"Loaded {len(self.raw_data)} image records.")

    def _get_drivable_polygons(self, labels: List[Dict]) -> Dict[str, List[Polygon]]:
        """Extracts drivable area polygons from a list of labels."""
        polygons = {"direct": [], "alternative": []}
        for label in labels:
            if label.get("category") == "drivable area":
                area_type = label.get("attributes", {}).get("areaType")
                if area_type in polygons and "poly2d" in label:
                    try:
                        # Ensure polygon has at least 3 points to be valid
                        if len(label["poly2d"][0]["vertices"]) >= 3:
                            poly = Polygon(label["poly2d"][0]["vertices"])
                            polygons[area_type].append(poly)
                    except Exception:
                        continue
        return polygons

    def _get_location_context(
        self, box: Dict, drivable_polys: Dict[str, List[Polygon]]
    ) -> str:
        """Determines if an object is on the main road, sidewalk, or off-road."""
        centroid = Point((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)
        for poly in drivable_polys.get("direct", []):
            if poly.contains(centroid):
                return "on_main_road"
        for poly in drivable_polys.get("alternative", []):
            if poly.contains(centroid):
                return "on_sidewalk"
        return "off_road"

    def _calculate_features(
        self, label: Dict, image_attrs: Dict
    ) -> Tuple[Optional[float], float, int]:
        """Calculates tightness_ratio, normalized_area, and difficulty_score."""
        box2d = label["box2d"]
        box_area = (box2d["x2"] - box2d["x1"]) * (box2d["y2"] - box2d["y1"])
        normalized_area = round(box_area / IMG_AREA, 6) if IMG_AREA > 0 else 0

        tightness_ratio = None
        if "poly2d" in label:
            try:
                # Ensure polygon has at least 3 points to be valid
                if len(label["poly2d"][0]["vertices"]) >= 3:
                    poly = Polygon(label["poly2d"][0]["vertices"])
                    if box_area > 0:
                        tightness_ratio = round(poly.area / box_area, 4)
            except Exception:
                tightness_ratio = None

        # Calculate difficulty score
        difficulty = 0
        attrs = label.get("attributes", {})
        if attrs.get("occluded"):
            difficulty += 1
        if attrs.get("truncated"):
            difficulty += 1
        if image_attrs.get("timeofday") == "night":
            difficulty += 1
        if image_attrs.get("weather") in ["rainy", "snowy", "foggy"]:
            difficulty += 1
        if tightness_ratio is not None and tightness_ratio < TIGHTNESS_THRESHOLD:
            difficulty += 1

        return tightness_ratio, normalized_area, difficulty

    def parse_to_dataframe(self) -> pd.DataFrame:
        """Parses the entire JSON file into a single, feature-rich DataFrame."""
        parsed_records = []
        for image_record in tqdm(self.raw_data, desc="Parsing"):
            if "labels" not in image_record:
                continue

            # Step 1: Extract contextual polygons for the whole image
            drivable_polys = self._get_drivable_polygons(image_record["labels"])

            # Step 2: Process each object label within this context
            for label in image_record["labels"]:
                category = label.get("category")
                if category not in DETECTION_CATEGORIES or "box2d" not in label:
                    continue

                location_context = self._get_location_context(
                    label["box2d"], drivable_polys
                )

                tightness, norm_area, difficulty = self._calculate_features(
                    label, image_record.get("attributes", {})
                )

                record = {
                    "image_name": image_record["name"],
                    "category": category,
                    "x1": label["box2d"]["x1"],
                    "y1": label["box2d"]["y1"],
                    "x2": label["box2d"]["x2"],
                    "y2": label["box2d"]["y2"],
                    "weather": image_record["attributes"]["weather"],
                    "scene": image_record["attributes"]["scene"],
                    "timeofday": image_record["attributes"]["timeofday"],
                    "occluded": label.get("attributes", {}).get("occluded", False),
                    "truncated": label.get("attributes", {}).get("truncated", False),
                    "traffic_light_color": label.get("attributes", {}).get(
                        "trafficLightColor", "none"
                    ),
                    "location_context": location_context,
                    "tightness_ratio": tightness,
                    "normalized_area": norm_area,
                    "difficulty_score": difficulty,
                }
                parsed_records.append(record)

        return pd.DataFrame(parsed_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dataframes for data analysis"
    )
    parser.add_argument(
        "--base-path", type=Path, required=True, help="Path to the bdd100k data"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path where the output parquet file will be saved.",
    )
    args = parser.parse_args()

    # Define paths to the specific files and directories
    image_path = args.base_path / "bdd100k_images_100k/bdd100k/images/100k"
    train_json_path = (
        args.base_path
        / "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    )
    val_json_path = (
        args.base_path
        / "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    )

    # Verify that the paths exist
    assert args.base_path.exists(), f"Base path not found: {args.base_path}"
    assert image_path.exists(), f"Image path not found: {image_path}"
    assert train_json_path.exists(), f"Train JSON not found: {train_json_path}"
    assert val_json_path.exists(), f"Validation JSON not found: {val_json_path}"

    print("All paths are correctly configured.")

    with open(train_json_path, "r") as f:
        train_data_raw = json.load(f)

    with open(val_json_path, "r") as f:
        val_data_raw = json.load(f)

    print(f"Total training samples: {len(train_data_raw)}")
    print(f"Total validation samples: {len(val_data_raw)}")

    object_classes = {}
    for image in train_data_raw:
        for label in image["labels"]:
            if "box2d" in label:
                if label["category"] in object_classes:
                    object_classes[label["category"]] += 1
                else:
                    object_classes[label["category"]] = 1

    DETECTION_CATEGORIES = list(object_classes.keys())

    parser_train = BddParser(train_json_path)
    df_train_granular = parser_train.parse_to_dataframe()

    parser_val = BddParser(val_json_path)
    df_val_granular = parser_val.parse_to_dataframe()

    df_train_granular.to_parquet(
        f"{str(args.output_path)}/extracted_data.pq", index=False
    )
    df_val_granular.to_parquet(
        f"{str(args.output_path)}/extracted_data_val.pq", index=False
    )
