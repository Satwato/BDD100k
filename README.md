# BDD Project
Link to Analysis: [Analysis](https://docs.google.com/document/d/17T18AqYnrkmfbcfiDiwpSvpI646yqBGhLBX1WHa2zzk/edit?usp=sharing)
## Prerequisites
Ensure you have Docker installed on your system.
Pull the repo and run 
```
docker build -t bdd-project-env
```
I tried finetuning RTDETR for 6 epochs: [Checkpoint](https://drive.google.com/file/d/1Ry-zMJ80B4OOYwW-UMTmZ7MeI3LYnIjy/view?usp=sharing)
Please put it under `train/checkpoints/` 

## Workflow Steps

### 1. Create DataFrames
Run the script to create dataframes for analysis:

```bash
docker run --rm \
  -v "<path to assignment_data_bdd>:/data" \
  -v "<path to local repo folder>:/bdd_files" \
    bdd-project-env \
    python data/create_dataframes.py \
    --base-path /data \
    --output-path /bdd_files
```

### 2. Run Streamlit for Analysis
Start the Streamlit app to perform analysis:

```bash
docker run -p 8501:8501 --rm \
  -v "<path to assignment_data_bdd>:/data" \
  -v "<path to where you pulled the repo>:/bdd_files" \
    bdd-project-env \
    streamlit run data/dashboard_with_val.py -- --train-df-path /bdd_files/extracted_data.pq --val-df-path /bdd_files/extracted_data_val.pq --base-data-path /data
```

### 3. Create COCO Files for Training
Generate the necessary COCO files:

```bash
docker run --rm \
  -v "<path to assignment_data_bdd>:/data" \
  -v "<path to where you pulled the repo>:/bdd_files" \
    bdd-project-env \
    python data/json_to_coco.py \
    --bdd-json-path /data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
    --output-json-path /data/bdd100k_images_100k/bdd100k/images/100k/train/_annotations.coco.json \
    --split-name train

docker run --rm \
  -v "<path to assignment_data_bdd>:/data" \
  -v "<path to where you pulled the repo>:/bdd_files" \
    bdd-project-env \
    python data/json_to_coco.py \
    --bdd-json-path /data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
    --output-json-path /data/bdd100k_images_100k/bdd100k/images/100k/val/_annotations.coco.json \
    --split-name val
```

### 4. Train the Model
Train the RT-DETR model:

```bash
docker run --rm \
  -v "<path to assignment_data_bdd>:/data" \
  -v "<path to where you pulled the repo>:/bdd_files" \
    bdd-project-env \
    python train/train_rtdetr.py
```

### 5. Evaluate the Model
Evaluate the trained model and generate metrics:

```bash
docker run --rm \
  -v "<path to assignment_data_bdd>:/data" \
  -v "<path to where you pulled the repo>:/bdd_files" \
    bdd-project-env \
    python train/eval_model.py --output-path /bdd_files
```

### 6. Visualize Results
Run the Streamlit app to visualize results and perform qualitative analysis:

```bash
docker run -p 8501:8501 --rm \
  -v "<path to assignment_data_bdd>:/data" \
  -v "<path to where you pulled the repo>:/bdd_files" \
    bdd-project-env \
    streamlit run eval/eval_dashboard.py -- --matched-predictions-path /bdd_files/results_iou.json --image-dir /data/bdd100k_images_100k/bdd100k/images/100k/val --processed-data-path /bdd_files/extracted_data_val.pq --coco-annotations-path /data/bdd100k_images_100k/bdd100k/images/100k/val/_annotations.coco.json
```

## Notes
- Ensure all paths are correctly set according to your project structure.
- Adjust the Docker volumes as necessary for your environment.
