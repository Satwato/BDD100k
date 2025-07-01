
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import AutoImageProcessor
from pycocotools.coco import COCO
from PIL import Image
from collections import Counter

class BDDDataset(Dataset):
    def __init__(self, img_dir, annotation_path, image_processor):
        super().__init__()
        self.img_dir = img_dir
        self.coco = COCO(annotation_path)
        self.image_processor = image_processor
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(path).convert("RGB")
        orig_size = torch.tensor(image.size[::-1])
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        annotations = self.coco.loadAnns(ann_ids)

        target = {'image_id': img_id, 'annotations': []}
        for ann in annotations:
            label = self.cat2label[ann['category_id']]
            target['annotations'].append({'bbox': ann['bbox'], 'category_id': label, 'area': ann['area']})

        processed_data = self.image_processor(images=image, annotations=target, return_tensors="pt")
        
        processed_data = {k: v[0] if k == 'labels' else v.squeeze(0) for k, v in processed_data.items()}
        
        processed_data['orig_size'] = orig_size
        processed_data['image_id'] = img_id
        processed_data['boxes_'] = target['annotations']
        return processed_data
    
    
class BDDDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.image_processor = AutoImageProcessor.from_pretrained(self.cfg.MODEL_CHECKPOINT, use_fast=True)
        self.class_names = []
        self.class_weights = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = BDDDataset(img_dir=self.cfg.TRAIN_IMG_DIR, annotation_path=self.cfg.TRAIN_ANNOTATION_PATH, image_processor=self.image_processor)
            self.val_dataset = BDDDataset(img_dir=self.cfg.VAL_IMG_DIR, annotation_path=self.cfg.VAL_ANNOTATION_PATH, image_processor=self.image_processor)
            self.class_names = [self.train_dataset.coco.cats[cat_id]['name'] for cat_id in self.train_dataset.cat_ids]
            
            print("Calculating class weights for loss function...")
            all_cat_ids = self.train_dataset.coco.getAnnIds()
            all_anns = self.train_dataset.coco.loadAnns(all_cat_ids)
            class_counts = Counter(ann['category_id'] for ann in all_anns)
            
            # Get counts in the same order as our cat2label mapping
            sorted_counts = torch.tensor([class_counts.get(cat_id, 1) for cat_id in self.train_dataset.cat_ids], dtype=torch.float32)
            
            # Calculate weights using inverse frequency. Add epsilon for stability.
            # The more frequent a class, the smaller its weight.
            total_samples = sorted_counts.sum()
            self.class_weights = total_samples / (len(self.class_names) * sorted_counts + 1e-6)
            
            print(f"Calculated Weights: {self.class_weights.tolist()}")
            

        print(f"Number of training examples: {len(self.train_dataset)}")
        print(f"Number of validation examples: {len(self.val_dataset)}")
        print(f"Class Names: {self.class_names}")

    def collate_fn(self, batch):
        # ... (no changes here)
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = [item['labels'] for item in batch]
        image_ids = [item['image_id'] for item in batch] 
        orig_sizes = torch.stack([item['orig_size'] for item in batch])
        orig_boxes = [item['boxes_'] for item in batch]
        pixel_mask = torch.stack([item['pixel_mask'] for item in batch]) if 'pixel_mask' in batch[0] else None
        return {'pixel_values': pixel_values, 'pixel_mask': pixel_mask, 'labels': labels, 'orig_sizes': orig_sizes, 'image_id':image_ids, 'orig_boxes': orig_boxes}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=True, num_workers=self.cfg.NUM_WORKERS, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=False, num_workers=self.cfg.NUM_WORKERS, collate_fn=self.collate_fn)
