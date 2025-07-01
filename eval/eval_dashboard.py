
import streamlit as st
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import json
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO


st.set_page_config(
    page_title="Qualitative Model Evaluation",
    layout="wide"
)

@st.cache_data
def load_processed_data(data_path: Path):
    """Loads the granular data analysis dataframe."""
    try:
        df = pd.read_parquet(data_path)
        df['gt_box_id'] = df.apply(lambda row: f"{row['image_name']}_{row['x1']:.2f}_{row['y1']:.2f}", axis=1)
        return df
    except FileNotFoundError:
        st.error(f"Processed data file not found at: {data_path}")
        return None

@st.cache_data
def load_matched_predictions(predictions_path: Path):
    """Loads matched model predictions from a JSON file."""
    try:
        with open(predictions_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Matched predictions file not found at: {predictions_path}")
        return None

@st.cache_data
def load_category_mapping(coco_path: Path):
    """Loads the category ID to name mapping from a COCO annotation file."""
    try:
        
        coco_json = COCO(str(coco_path).replace("valid", "train"))
        cat_ids = sorted(coco_json.getCatIds())
        class_names = [coco_json.cats[cat_id]['name'] for cat_id in cat_ids]
        id2label = {i: name for i, name in enumerate(class_names)}
        return id2label
    except FileNotFoundError:
        st.error(f"COCO annotation file not found at: {coco_path}")
        return None

@st.cache_data
def create_evaluation_dataframe(predictions, df_gt, id2cat):
    """
    Creates a single, filterable DataFrame starting from the ground truth
    and augmenting it with prediction data. This ensures all ground truth
    objects, including missed ones (False Negatives), are included.
    """
    
    cat2id = {v: k for k, v in id2cat.items()}
    df_gt['category_id'] = df_gt['category'].map(cat2id)

    pred_lookup = {}
    for img_name, gt_list in predictions.items():
        for gt_obj in gt_list:
            if gt_obj['is_detected']:
                box = gt_obj['gt_box']
                gt_box_id = f"{img_name}_{box[0]:.2f}_{box[1]:.2f}"
                pred_lookup[gt_box_id] = gt_obj

    def get_detection_info(row):
        gt_box_id = row['gt_box_id']
        match = pred_lookup.get(gt_box_id)
        if match:
            is_misclassified = row['category_id'] != match['matched_pred']['pred_label']
            return True, is_misclassified, match['matched_pred']
        else:
            return False, False, None

    detection_info = df_gt.apply(get_detection_info, axis=1)
    df_gt[['is_detected', 'is_misclassified', 'matched_pred']] = pd.DataFrame(detection_info.tolist(), index=df_gt.index)
    
    return df_gt

def draw_on_image(image_path: Path, gt_box_to_highlight, matched_pred, id2cat):
    """
    Draws a specific ground truth box and its matched prediction.
    - Bright Green: The Ground Truth box of interest.
    - Red: The matched prediction (if it exists).
    """
    if not image_path.exists():
        st.warning(f"Image not found at {image_path}")
        return None
    
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gx1, gy1, gx2, gy2 = int(gt_box_to_highlight['x1']), int(gt_box_to_highlight['y1']), int(gt_box_to_highlight['x2']), int(gt_box_to_highlight['y2'])
    cv2.rectangle(image, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3)
    gt_label_name = id2cat.get(gt_box_to_highlight['category_id'], 'N/A')
    gt_text = f"GT: {gt_label_name}"
    cv2.putText(image, gt_text, (gx1, gy1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if matched_pred:
        pred_box = matched_pred['pred_box']
        px1, py1, px2, py2 = map(int, pred_box)
        cv2.rectangle(image, (px1, py1), (px2, py2), (255, 0, 0), 2)
        
        pred_label_name = id2cat.get(matched_pred['pred_label'], 'N/A')
        score = matched_pred['score']
        iou = matched_pred['iou']
        
        pred_text = f"Pred: {pred_label_name} ({score:.2f})"
        cv2.putText(image, pred_text, (px1, py2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if gt_label_name == pred_label_name:
            st.success(f"Correct Classification: Predicted '{pred_label_name}' with IoU: {iou:.2f}")
        else:
            st.error(f"Misclassification: Predicted '{pred_label_name}' instead of '{gt_label_name}'. IoU: {iou:.2f}")
    else:
        st.warning("False Negative: No prediction was matched to this ground truth object.")

    return image


st.title("Qualitative Model Evaluation Dashboard")

# --- Command Line Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--matched-predictions-path", type=Path, required=True)
parser.add_argument("--image-dir", type=Path, required=True)
parser.add_argument("--processed-data-path", type=Path, required=True)
parser.add_argument("--coco-annotations-path", type=Path, required=True)
try:
    args = parser.parse_args()
except SystemExit:
    st.error("Please provide all required command line arguments.")
    st.stop()


# --- Load Data ---
df_val_granular = load_processed_data(args.processed_data_path)
predictions = load_matched_predictions(args.matched_predictions_path)
id2cat = load_category_mapping(args.coco_annotations_path)

if df_val_granular is None or predictions is None or id2cat is None:
    st.stop()

# Create the main dataframe for evaluation
df_eval = create_evaluation_dataframe(predictions, df_val_granular, id2cat)


# --- Sidebar and Page Navigation ---
st.sidebar.title("Evaluation Dashboard")
page = st.sidebar.radio("Choose a Page", ["Qualitative Explorer", "Error Analysis"])
st.sidebar.markdown("---")


# --- Page 1: Qualitative Explorer ---
if page == "Qualitative Explorer":
    st.header("Qualitative Sample Explorer")
    st.info("Visually inspect model performance on specific slices of data. Use the filters to find interesting cases.")

    with st.sidebar.form(key='filter_form'):
        st.header("Analysis Filters")
        
        valid_categories = df_eval['category'].dropna().unique()
        filter_category = st.selectbox("Object Category", sorted(valid_categories))

        filter_detection_status = st.selectbox(
            "Detection Status", 
            ("Any", "Correctly Detected (TP)", "Misclassified", "Missed (FN)"), 
            index=0
        )
        st.markdown("---")

        filter_scene = st.multiselect("Scene", sorted(df_eval['scene'].dropna().unique()), default=None)
        filter_weather = st.multiselect("Weather", sorted(df_eval['weather'].dropna().unique()), default=None)
        filter_timeofday = st.multiselect("Time of Day", sorted(df_eval['timeofday'].dropna().unique()), default=None)
        filter_occluded = st.selectbox("Is Occluded?", ("Any", "Yes", "No"), index=0)
        filter_truncated = st.selectbox("Is Truncated?", ("Any", "Yes", "No"), index=0)
        min_difficulty = st.slider("Minimum Difficulty Score", 0, 5, 0)

        submitted = st.form_submit_button("Apply Filters & Load Sample")

    if submitted:
        filtered_df = df_eval.copy()
        if filter_category:
            filtered_df = filtered_df[filtered_df['category'] == filter_category]
        if filter_scene:
            filtered_df = filtered_df[filtered_df['scene'].isin(filter_scene)]
        if filter_weather:
            filtered_df = filtered_df[filtered_df['weather'].isin(filter_weather)]
        if filter_timeofday:
            filtered_df = filtered_df[filtered_df['timeofday'].isin(filter_timeofday)]
        if filter_occluded != "Any":
            filtered_df = filtered_df[filtered_df['occluded'] == (filter_occluded == "Yes")]
        if filter_truncated != "Any":
            filtered_df = filtered_df[filtered_df['truncated'] == (filter_truncated == "Yes")]
        if min_difficulty > 0:
            filtered_df = filtered_df[filtered_df['difficulty_score'] >= min_difficulty]

        if filter_detection_status == "Correctly Detected (TP)":
            filtered_df = filtered_df[(filtered_df['is_detected'] == True) & (filtered_df['is_misclassified'] == False)]
        elif filter_detection_status == "Misclassified":
            filtered_df = filtered_df[filtered_df['is_misclassified'] == True]
        elif filter_detection_status == "Missed (FN)":
            filtered_df = filtered_df[filtered_df['is_detected'] == False]

        st.info(f"Found **{len(filtered_df)}** ground truth objects matching your criteria.")

        if not filtered_df.empty:
            random_sample = filtered_df.sample(1).iloc[0]
            st.session_state['current_sample'] = random_sample.to_dict()
        else:
            st.warning("No samples found for the selected filters.")
            if 'current_sample' in st.session_state:
                del st.session_state['current_sample']

    if 'current_sample' in st.session_state:
        sample = st.session_state['current_sample']
        image_name = sample['image_name']
        image_path = args.image_dir / image_name
        
        st.subheader(f"Image: `{image_name}`")
        st.write(f"**Context:** {sample.get('weather', 'N/A')}, {sample.get('scene', 'N/A')}, {sample.get('timeofday', 'N/A')}")
        st.write(f"**Difficulty Score:** {sample.get('difficulty_score', 'N/A')}")
        
        matched_pred = sample.get('matched_pred')
        if isinstance(matched_pred, float) and np.isnan(matched_pred):
            matched_pred = None

        image_with_boxes = draw_on_image(image_path, sample, matched_pred, id2cat)
        
        if image_with_boxes is not None:
            st.image(image_with_boxes, use_column_width=True)

# --- Page 2: Error Analysis ---
elif page == "Error Analysis":
    st.header("Error Analysis Statistics")
    st.info("This page provides a quantitative breakdown of the model's most common failure modes.")

    valid_categories = df_eval['category'].dropna().unique()
    analysis_category = st.selectbox("Select a Ground Truth Category to Analyze:", sorted(valid_categories))

    if analysis_category:
        df_cat = df_eval[df_eval['category'] == analysis_category]

        st.markdown("---")
        st.subheader(f"Analysis for Ground Truth: `{analysis_category}`")

        # --- Overall Detection Status Chart ---
        st.markdown("#### Overall Detection Status")
        
        total_gt = len(df_cat)
        missed_count = len(df_cat[df_cat['is_detected'] == False])
        misclassified_count = len(df_cat[df_cat['is_misclassified'] == True])
        correct_count = len(df_cat[(df_cat['is_detected'] == True) & (df_cat['is_misclassified'] == False)])
        
        status_data = {
            'Correctly Detected': correct_count,
            'Misclassified': misclassified_count,
            'Missed (FN)': missed_count
        }
        
        status_df = pd.DataFrame(list(status_data.items()), columns=['Status', 'Count'])
        
        fig, ax = plt.subplots()
        sns.barplot(data=status_df, x='Status', y='Count', ax=ax)
        ax.set_title(f"Detection Status for '{analysis_category}' (Total: {total_gt})")
        st.pyplot(fig)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)

        with col1:
            # --- Misclassification Analysis ---
            st.markdown("#### Misclassification Errors")
            misclassified_df = df_cat[df_cat['is_misclassified'] == True]
            if not misclassified_df.empty:
                # Extract the predicted label name
                misclassified_df['pred_category_name'] = misclassified_df['matched_pred'].apply(
                    lambda x: id2cat.get(x['pred_label']) if x else None
                )
                confusion_counts = misclassified_df['pred_category_name'].value_counts()
                
                fig, ax = plt.subplots()
                sns.barplot(x=confusion_counts.index, y=confusion_counts.values, ax=ax)
                ax.set_title(f"What '{analysis_category}' is commonly confused with:")
                ax.set_ylabel("Number of Misclassifications")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
            else:
                st.success(f"No misclassification errors found for '{analysis_category}'.")

        with col2:
            # --- False Negative Analysis ---
            st.markdown("#### Missed Detections (False Negatives)")
            missed_df = df_cat[df_cat['is_detected'] == False]
            if not missed_df.empty:
                st.write(f"**Characteristics of the {len(missed_df)} missed '{analysis_category}' objects:**")
                
                # Show distribution of difficulty for missed objects
                fig, ax = plt.subplots()
                sns.countplot(data=missed_df, x='difficulty_score', ax=ax)
                ax.set_title("Difficulty Score of Missed Objects")
                st.pyplot(fig)
                
                # Show scene distribution for missed objects
                fig, ax = plt.subplots()
                missed_df['scene'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
                ax.set_title("Scene Distribution of Missed Objects")
                ax.set_ylabel('')
                st.pyplot(fig)
            else:
                st.success(f"No missed detections found for '{analysis_category}'.")
