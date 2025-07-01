
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection
from pycocotools.coco import COCO
from PIL import Image, ImageDraw, ImageFont
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.detection import MeanAveragePrecision
from collections import Counter
import warnings
import torchmetrics
from tqdm import tqdm 
import pandas as pd
import traceback
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
from data_utils import BDDDataModule
from train_rtdetr import RTDETR_FineTuner
import json
warnings.filterwarnings("ignore")


class CFG:
    MODEL_CHECKPOINT = "PekingU/rtdetr_r18vd"
    TRAIN_ANNOTATION_PATH = "/data/bdd100k_images_100k/bdd100k/images/100k/train/_annotations.coco.json"
    VAL_ANNOTATION_PATH = "/data/bdd100k_images_100k/bdd100k/images/100k/val/_annotations.coco.json"
    TRAIN_IMG_DIR = "/data/bdd100k_images_100k/bdd100k/images/100k/train/"
    VAL_IMG_DIR = "/data/bdd100k_images_100k/bdd100k/images/100k/val/"
    TRAINED_MODEL_PATH = "/bdd_files/train/checkpoints/rtdetr-bdd-best-epoch=06-val_loss=5.1101.ckpt"
    PROCESSED_DATA_PATH = "/bdd_files"
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    

def draw_boxes(image, boxes, labels, scores, color_map):
    """Draws bounding boxes on an image."""
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        color = color_map.get(label, "white")
        draw.rectangle(box, outline=color, width=3)
        text = f"{label}: {score:.2f}"
        text_bbox = draw.textbbox((box[0], box[1] - 15), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box[0], box[1] - 15), text, fill="black", font=font)
    return img_with_boxes

def visualize_predictions(config, model, data_module, num_images=4):
    """Loads data, runs inference, and plots the results."""
    val_loader = data_module.val_dataloader()
    id2label = {i: name for i, name in enumerate(data_module.class_names)}
    
    colors = plt.cm.get_cmap('hsv', len(data_module.class_names))
    color_map = {name: tuple(int(c * 255) for c in colors(i)[:3]) for i, name in enumerate(data_module.class_names)}

    batch = next(iter(val_loader))
    pixel_values = batch["pixel_values"]
    orig_sizes = batch["orig_sizes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values.to(device))

    results = data_module.image_processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=orig_sizes
    )

    fig, axes = plt.subplots(num_images, 2, figsize=(20, 8 * num_images))
    fig.suptitle("Model Predictions vs. Ground Truth", fontsize=20)

    for i in range(min(num_images, config.BATCH_SIZE)):
        # preds
        pred_scores = results[i]["scores"].cpu().tolist()
        pred_labels = [id2label[label.item()] for label in results[i]["labels"].cpu()]
        pred_boxes = results[i]["boxes"].cpu().tolist()

        # ground truth
        img_id = batch['image_id'][i]
        img_info = data_module.val_dataset.coco.loadImgs(img_id)[0]
        original_image = Image.open(os.path.join(config.VAL_IMG_DIR, img_info['file_name'])).convert("RGB")

        ann_ids = data_module.val_dataset.coco.getAnnIds(imgIds=img_id)
        annotations = data_module.val_dataset.coco.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        trgs = []
        for ann in annotations:
            box = ann['bbox']
            gt_boxes.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
            cat_id = ann['category_id']
            gt_labels.append(data_module.val_dataset.coco.cats[cat_id]['name'])

        img_pred = draw_boxes(original_image, pred_boxes, pred_labels, pred_scores, color_map)
        axes[i, 0].imshow(img_pred)
        axes[i, 0].set_title(f"Predicted Boxes (Image ID: {img_id})")
        axes[i, 0].axis("off")

        img_gt = draw_boxes(original_image, gt_boxes, gt_labels, [1.0] * len(gt_labels), color_map)
        axes[i, 1].imshow(img_gt)
        axes[i, 1].set_title(f"Ground Truth Boxes (Image ID: {img_id})")
        axes[i, 1].axis("off")
        

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('eval.png')

def ann_to_target(annotations):
    """
    Converts annotations to target format for model training.
    """
    new_bboxes = []
    label_tensor = []
    for box in annotations:
        x1,y1, w, h = box['bbox']
        new_bboxes.append(torch.Tensor([x1,y1,x1+w, y1+h]))
        label_tensor.append(box['category_id'])
    return torch.stack(new_bboxes), torch.Tensor(label_tensor).int()
        
    
def get_preds(model, data_module):
    """
    Calculates and prints the mean Average Precision (mAP) for the model
    on the entire validation dataset.
    """
    print("\nStarting mAP calculation on the validation set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    
    val_loader = data_module.val_dataloader()
    full_preds, full_targets = [], []
    for batch in tqdm(val_loader, desc="Calculating mAP"):
        pixel_values = batch["pixel_values"].to(device)
        orig_sizes = batch["orig_sizes"]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        # get bounding boxes processed from preds
        preds = data_module.image_processor.post_process_object_detection(
            outputs, threshold=0.2, target_sizes=orig_sizes
        )
        targets = []
        for i in range(len(batch["labels"])):
            img_info = data_module.val_dataset.coco.loadImgs(batch['image_id'][i])[0]['file_name']
            boxes, labels = ann_to_target(batch["orig_boxes"][i])
            targets.append({
                'image_name': img_info,
                'boxes':boxes,
                'labels': labels
                
            })
        preds = [{k: v.detach().cpu() if hasattr(v, 'detach') else v for k, v in d.items()} for d in preds]
        full_preds.extend(preds)
        full_targets.extend(targets)
    return full_preds, full_targets

def calculate_mAP(preds, targets, data_module):
    map_metric = torchmetrics.detection.MeanAveragePrecision(class_metrics=True)
    map_metric.update(preds, targets)
    id2label = {i: name for i, name in enumerate(data_module.class_names)}
    
    print("\n" + "="*50)
    print("mAP Calculation Complete.")
    print("="*50)
    try:
        map_results = map_metric.compute()
    
        print(f"\nOverall mAP: {map_results['map'].item():.4f}")
        print(f"Overall mAP@.50 (PASCAL VOC): {map_results['map_50'].item():.4f}")
        print(f"Overall mAP@.75 (Strict): {map_results['map_75'].item():.4f}")

        print("\n--- Average Precision (AP) per Class ---")
        map_per_class = map_results.get('map_per_class', [])
        
        # if map_per_class exists
        if isinstance(map_per_class, torch.Tensor):
            if map_per_class.numel() > 0:  # Check if tensor is not empty
                for i, ap in enumerate(map_per_class):
                    class_name = id2label.get(i, f"Unknown Class {i}")
                    print(f"  - {class_name:<15}: {ap.item():.4f}")
            else:
                print("No per-class AP available (empty tensor).")
        
        elif isinstance(map_per_class, list) and len(map_per_class) > 0:
            for i, ap in enumerate(map_per_class):
                class_name = id2label.get(i, f"Unknown Class {i}")
                ap_value = ap.item() if hasattr(ap, 'item') else ap
                print(f"  - {class_name:<15}: {ap_value:.4f}")
        else:
            print("Could not retrieve per-class AP. Check if 'class_metrics=True' is set.")
        
        print("\n--- Average Recall (AR) per Class ---")
        mar_per_class = map_results.get('mar_100_per_class', [])
        
        if isinstance(mar_per_class, torch.Tensor):
            if mar_per_class.numel() > 0:  # Check if tensor is not empty
                for i, ar in enumerate(mar_per_class):
                    class_name = id2label.get(i, f"Unknown Class {i}")
                    print(f"  - {class_name:<15}: {ar.item():.4f}")
            else:
                print("No per-class AR available (empty tensor).")
        elif isinstance(mar_per_class, list) and len(mar_per_class) > 0:
            for i, ar in enumerate(mar_per_class):
                class_name = id2label.get(i, f"Unknown Class {i}")
                
                #incase of scalar values
                ar_value = ar.item() if hasattr(ar, 'item') else ar
                print(f"  - {class_name:<15}: {ar_value:.4f}")
        else:
            print("Could not retrieve per-class AR.")

        print("\n" + "="*50)
        return map_results, map_metric
    except Exception as e:
       print("ERROR", traceback.format_exc())


def match_predictions(preds, targets, iou_threshold=0.5):
    """
    Matches predictions to ground truth targets for a single image.
    Returns a list of all GT boxes, with matched prediction info attached.
    """
    gt_boxes = targets['boxes']
    gt_labels = targets['labels']
    
    pred_boxes = preds['boxes']
    pred_labels = preds['labels']
    pred_scores = preds['scores']

    results = []
    if len(pred_boxes) == 0:
        for i, gt_box in enumerate(gt_boxes):
            results.append({
                "gt_box": gt_box.tolist(), "gt_label": gt_labels[i].item(),
                "is_detected": False, "matched_pred": None
            })
        return results

    if len(gt_boxes) == 0:
        return []

    iou_matrix = box_iou(gt_boxes, pred_boxes)
    
    #find the best prediction for each ground truth box
    best_pred_iou, best_pred_idx = iou_matrix.max(dim=1)
    
    for i, gt_box in enumerate(gt_boxes):
        gt_label = gt_labels[i].item()
        iou = best_pred_iou[i].item()
        
        match_info = {
            "gt_box": gt_box.tolist(), "gt_label": gt_label,
            "is_detected": False, "matched_pred": None
        }

        if iou >= iou_threshold:
            pred_idx = best_pred_idx[i].item()
            match_info["is_detected"] = True
            match_info["matched_pred"] = {
                "pred_box": pred_boxes[pred_idx].tolist(),
                "pred_label": pred_labels[pred_idx].item(),
                "score": pred_scores[pred_idx].item(),
                "iou": iou
            }
        results.append(match_info)
        
    return results

def stratified_metrics(image_to_context, all_targets, all_preds, id2label):
    strata = ['scene', 'weather', 'timeofday']
    for stratum in strata:
        unique_values = image_to_context[stratum].unique()
        for value in unique_values:
            # Filter predictions and targets for this specific stratum value
            stratified_preds, stratified_targets = [], []
            image_names_in_stratum = image_to_context[image_to_context[stratum] == value].index
            
            for i, target in enumerate(all_targets):
                img_id = target['image_id'].item()
                img_name = data_module.val_dataset.coco.loadImgs(img_id)[0]['file_name']
                if img_name in image_names_in_stratum:
                    stratified_preds.append(all_preds[i])
                    stratified_targets.append(target)
            
            if not stratified_preds:
                continue

            stratum_map_metric = torchmetrics.detection.MeanAveragePrecision(class_metrics=True)
            stratum_map_metric.update(stratified_preds, stratified_targets)
            stratum_map_results = stratum_map_metric.compute()
            print_mAP_results(stratum_map_results, id2label, title=f"mAP for {stratum} = {value}")


def print_mAP_results(map_results, id2label, title="mAP Results"):
    """Prints a formatted summary of mAP results."""
    print("\n" + "="*20 + f" {title} " + "="*20)
    print(f"  Overall mAP@.50:.95: {map_results['map'].item():.4f}")
    print(f"  Overall mAP@.50:      {map_results['map_50'].item():.4f}")
    print(f"  Overall mAP@.75:      {map_results['map_75'].item():.4f}")
    
    print("\n  --- AP per Class ---")
    for i, ap in enumerate(map_results.get('map_per_class', [])):
        class_id = map_results['classes'][i].item()
        class_name = id2label.get(class_id, f"ID:{class_id}")
        print(f"    - {class_name:<15}: {ap.item():.4f}")
    print("="* (42 + len(title)))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate dataframes for data analysis"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path where the output parquet file will be saved.",
    )
    args = parser.parse_args()

    cfg = CFG
    try:
        model = RTDETR_FineTuner.load_from_checkpoint(cfg.TRAINED_MODEL_PATH)
        print(f"Successfully loaded trained model from {cfg.TRAINED_MODEL_PATH}")
        data_module = BDDDataModule(cfg)
        data_module.setup()
        preds, targets = get_preds(model, data_module)
        val_loader = data_module.val_dataloader()
        id2label = data_module.val_dataset.id2cat
        map_result, map_metric = calculate_mAP(preds, targets, data_module)
        df_granular = pd.read_parquet(f"{cfg.PROCESSED_DATA_PATH}extracted_data_val.pq")
        image_to_context = df_granular.drop_duplicates('image_name').set_index('image_name')
        stratified_metrics(image_to_context, targets, preds, id2label)
        all_matched_predictions_for_json = {}
        for pred, target in tqdm(zip(preds, targets)):
            matched_results = match_predictions(pred, target)
            all_matched_predictions_for_json[target['image_name']] = matched_results
        with open(f"{str(args.output_path)}/results_iou.json", 'w') as f:
            json.dump(all_matched_predictions_for_json, f, indent=4)
    except Exception as e:
        print(traceback.format_exc())