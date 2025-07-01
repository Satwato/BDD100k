
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection
from pycocotools.coco import COCO
from PIL import Image
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.detection import MeanAveragePrecision
from collections import Counter
import warnings
import torchmetrics
from tqdm import tqdm 
import traceback
import matplotlib.pyplot as plt
from data_utils import BDDDataModule
warnings.filterwarnings("ignore")

class CFG:
    MODEL_CHECKPOINT = "PekingU/rtdetr_r18vd"
    TRAIN_ANNOTATION_PATH = "/Users/satwato.dey/Documents/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/train/_annotations.coco.json"
    VAL_ANNOTATION_PATH = "/Users/satwato.dey/Documents/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/valid/_annotations.coco.json"
    TRAIN_IMG_DIR = "/Users/satwato.dey/Documents/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/train"
    VAL_IMG_DIR = "/Users/satwato.dey/Documents/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/valid"
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 10

class RTDETR_FineTuner(pl.LightningModule):
    def __init__(self, config, num_classes, class_names, class_weights=None):
        super().__init__()
        self.cfg = config
        self.class_names = class_names
        
        
        model_config = AutoConfig.from_pretrained(
            self.cfg.MODEL_CHECKPOINT,
            num_labels=num_classes,
        )
        
        
        if class_weights is not None:
            print("Applying custom class weights to the model's configuration.")
            # The 'alpha' parameter of the Focal Loss is used for class weighting
            model_config.focal_alpha = class_weights.tolist() # Must be a list for the config
            
        
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.cfg.MODEL_CHECKPOINT,
            config=model_config,
            ignore_mismatched_sizes=True # Still needed to replace the head
        )

        self.val_map_metric = MeanAveragePrecision(box_format="cxcywh", class_metrics=True)
        self.save_hyperparameters()

    def forward(self, pixel_values, pixel_mask=None):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def training_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch['pixel_values'], pixel_mask=batch.get('pixel_mask'), labels=batch['labels'])
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch['pixel_values'], pixel_mask=batch.get('pixel_mask'), labels=batch['labels'])
        loss = outputs.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
   

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.LEARNING_RATE, weight_decay=self.cfg.WEIGHT_DECAY)
        return optimizer


if __name__ == '__main__':
    
    pl.seed_everything(42)
    data_module = BDDDataModule(CFG)
    data_module.setup() 
    
    model = RTDETR_FineTuner(
        CFG, 
        num_classes=len(data_module.class_names),
        class_names=data_module.class_names,
        class_weights=data_module.class_weights
    )
    

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, dirpath='checkpoints/', filename='rtdetr-bdd-best-{epoch:02d}-{val_loss:.4f}')
    trainer = pl.Trainer(max_epochs=CFG.MAX_EPOCHS, accelerator='auto', devices='auto', callbacks=[checkpoint_callback], precision="16-mixed", limit_train_batches=10, limit_val_batches=10)
    
    print("Starting fine-tuning...")
    trainer.fit(model, data_module)
    print("Fine-tuning finished.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")