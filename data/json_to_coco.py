import json
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_bdd_to_coco(bdd_json_path: Path, output_path: Path, split_name: str):
    """
    Main conversion function.

    Args:
        bdd_json_path: Path to the BDD100k source JSON file.
        output_path: Path to save the output COCO JSON file.
        split_name: The name of the split ('train' or 'val').
    """
    print(f"Loading BDD annotations from: {bdd_json_path}")
    with open(bdd_json_path, "r") as f:
        bdd_data = json.load(f)

    coco_output = {
        "info": {
            "description": f"BDD100k Object Detection Dataset - {split_name} split",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    category_names = [
        "traffic light",
        "traffic sign",
        "car",
        "person",
        "bus",
        "truck",
        "rider",
        "bike",
        "motor",
        "train",
    ]

    for i, cat_name in enumerate(category_names):
        coco_output["categories"].append(
            {"id": i + 1, "name": cat_name, "supercategory": "object"}
        )

    bdd_cat_to_coco_id = {name: i + 1 for i, name in enumerate(category_names)}

    image_id_counter = 0
    annotation_id_counter = 0

    print(f"Processing {len(bdd_data)} images for the '{split_name}' split...")
    for img_record in tqdm(bdd_data, desc=f"Converting {split_name}"):
        image_info = {
            "id": image_id_counter,
            "file_name": f"{img_record['name']}",
            "height": 720,
            "width": 1280,
            "license": None,
            "coco_url": None,
            "date_captured": None,
        }
        coco_output["images"].append(image_info)

        if "labels" in img_record:
            for label in img_record["labels"]:
                if "box2d" in label and label["category"] in bdd_cat_to_coco_id:
                    box2d = label["box2d"]
                    x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
                    width = x2 - x1
                    height = y2 - y1

                    bbox = [x1, y1, width, height]
                    area = width * height

                    annotation_info = {
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": bdd_cat_to_coco_id[label["category"]],
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                    coco_output["annotations"].append(annotation_info)
                    annotation_id_counter += 1

        image_id_counter += 1

    print(f"Conversion complete. Saving COCO annotations to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_output, f)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BDD100k annotations to COCO format."
    )
    parser.add_argument(
        "--bdd-json-path",
        type=Path,
        required=True,
        help="Path to the source BDD100k JSON file.",
    )
    parser.add_argument(
        "--output-json-path",
        type=Path,
        required=True,
        help="Path to save the output COCO JSON file.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        required=True,
        choices=["train", "valid"],
        help="Name of the data split (e.g., 'train', 'valid').",
    )
    args = parser.parse_args()

    convert_bdd_to_coco(args.bdd_json_path, args.output_json_path, args.split_name)
