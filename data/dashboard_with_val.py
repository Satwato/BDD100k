# data_analysis/src/dashboard.py

import argparse
import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="BDD100k Granular Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Caching Data Loading & Feature Engineering ---
@st.cache_data
def load_and_prep_data(file_path: Path) -> pd.DataFrame:
    """Loads the processed parquet file and engineers all necessary features."""
    try:
        df = pd.read_parquet(file_path)

        # geometric features
        df["width"] = df["x2"] - df["x1"]
        df["height"] = df["y2"] - df["y1"]
        df["aspect_ratio"] = df["width"] / (df["height"] + 1e-6)
        df["centroid_x"] = df["x1"] + df["width"] / 2
        df["centroid_y"] = df["y1"] + df["height"] / 2
        df["normalized_area"] = (df["width"] * df["height"]) / (1280 * 720)

        df["difficulty_score"] = (
            df["occluded"].astype(int)
            + df["truncated"].astype(int)
            + (df["timeofday"] == "night").astype(int)
            + (df["weather"].isin(["rainy", "snowy", "foggy"])).astype(int)
        )
        return df
    except FileNotFoundError:
        return None


# --- Plotting Functions ---
def plot_metric_distribution(
    df: pd.DataFrame, metric: str, title: str, log_scale_y: bool = False, bins: int = 30
):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[metric].dropna(), kde=True, ax=ax, bins=bins)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Count")
    if log_scale_y:
        ax.set_yscale("symlog")
    st.pyplot(fig)


def plot_log_transformed_distribution(df: pd.DataFrame, metric: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    log_transformed_data = np.log1p(df[metric].dropna())
    sns.histplot(log_transformed_data, kde=True, ax=ax, bins=40)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(f"Log(1 + {metric.replace('_', ' ').title()})")
    ax.set_ylabel("Count")
    st.pyplot(fig)


def plot_categorical_distribution(df: pd.DataFrame, metric: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    df[metric].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax, startangle=90)
    ax.set_title(title, fontsize=16)
    ax.legend()
    ax.set_ylabel("")
    st.pyplot(fig)


def draw_boxes_on_image(image_path: Path, boxes_df: pd.DataFrame):
    if not image_path.exists():
        st.warning(f"Image not found at {image_path}")
        return None

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    categories = boxes_df["category"].unique()
    colors = [
        tuple(int(c * 255) for c in plt.cm.viridis(i / len(categories))[:3])
        for i in range(len(categories))
    ]
    color_map = dict(zip(categories, colors))

    for _, row in boxes_df.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        category = row["category"]
        color = color_map.get(category, (255, 0, 0))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{category} (Diff: {row.get('difficulty_score', 'N/A')})"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img

parser = argparse.ArgumentParser()
parser.add_argument("--train-df-path", type=Path, required=True)
parser.add_argument("--val-df-path", type=Path, required=True)
parser.add_argument("--base-data-path", type=Path, required=True)
args = parser.parse_args()

TRAIN_DATA_PATH = args.train_df_path
VAL_DATA_PATH = args.val_df_path

df_train = load_and_prep_data(TRAIN_DATA_PATH)
df_val = load_and_prep_data(VAL_DATA_PATH)

st.sidebar.title("BDD100k Granular Analysis")

# Dataset Selector
st.sidebar.markdown("### Select Dataset")
dataset_choice = st.sidebar.selectbox(
    "Choose a dataset split to analyze:",
    ("Train", "Validation", "Train vs. Val Comparison"),
)
st.sidebar.markdown("---")

# determine which dataframe to use based on choice
df_display = None
image_dir_display = None

if dataset_choice == "Train":
    df_display = df_train
    image_dir_display = (
        args.base_data_path
        / "bdd100k_images_100k"
        / "bdd100k"
        / "images"
        / "100k"
        / "train"
    )
elif dataset_choice == "Validation":
    df_display = df_val
    image_dir_display = (
        args.base_data_path
        / "bdd100k_images_100k"
        / "bdd100k"
        / "images"
        / "100k"
        / "val"
    )
else:
    df_display = None

# Main page navigation
page_options = [
    "Dataset Overview",
    "Deep Dive by Category",
    "Contextual Analysis",
    "Qualitative Sample Explorer",
]
if dataset_choice == "Train vs. Val Comparison":
    page_options = ["Train vs. Val Comparison"]

page = st.sidebar.radio("Choose a Page", page_options)
st.sidebar.markdown("---")


# --- Page Rendering ---

if dataset_choice == "Train vs. Val Comparison":
    st.header("Train vs. Validation Set Comparison")
    st.info(
        "This page compares the key distributions between the training and validation sets to check for potential distribution shift."
    )

    if df_train is not None and df_val is not None:
        df_train_copy = df_train.copy()
        df_val_copy = df_val.copy()
        df_train_copy["split"] = "Train"
        df_val_copy["split"] = "Validation"
        combined_df = pd.concat([df_train_copy, df_val_copy], ignore_index=True)

        st.subheader("Class Distribution Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(
            data=combined_df,
            y="category",
            hue="split",
            ax=ax,
            order=combined_df["category"].value_counts().index,
        )
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Environmental Context Comparison")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots()
            sns.countplot(data=combined_df, x="timeofday", hue="split", ax=ax)
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.countplot(data=combined_df, x="weather", hue="split", ax=ax)
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        with col3:
            fig, ax = plt.subplots()
            sns.countplot(data=combined_df, x="scene", hue="split", ax=ax)
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error(
            "Could not load both train and validation data. Please ensure both processed files exist."
        )


elif df_display is not None:
    if page == "Dataset Overview":
        st.header(f"Dataset Overview ({dataset_choice})")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Labeled Objects", f"{len(df_display):,}")
        col2.metric("Unique Images", f"{df_display['image_name'].nunique():,}")
        col3.metric("Number of Categories", df_display["category"].nunique())

        st.markdown("### Overall Class Distribution")
        fig, ax = plt.subplots(figsize=(12, 5))
        df_display["category"].value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of Instances")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

        st.markdown("### Environmental Context")
        col_env1, col_env2 = st.columns(2)
        with col_env1:
            plot_categorical_distribution(
                df_display, "timeofday", "Distribution by Time of Day"
            )
        with col_env2:
            plot_categorical_distribution(
                df_display, "weather", "Distribution by Weather"
            )

        # --- NEW: Data Quality Section ---
        st.markdown("---")
        st.header("Data Quality & Annotation Issues")
        st.info(
            "This section examines the raw dimensions of the bounding boxes to identify potential annotation errors or challenging, very small objects."
        )

        col_quality1, col_quality2 = st.columns(2)
        with col_quality1:
            # It's more informative to see the distribution of very small boxes
            fig, ax = plt.subplots(figsize=(10, 5))
            small_widths = df_display[df_display["width"] < 20]["width"]
            sns.histplot(small_widths, ax=ax, bins=20, kde=True)
            ax.set_title("Distribution of Small Bounding Box Widths (<20px)")
            ax.set_xlabel("Width (pixels)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Add stats on very small boxes
            num_lt_5px = (df_display["width"] < 5).sum()
            st.metric(label="Boxes with Width < 5 pixels", value=f"{num_lt_5px:,}")

        with col_quality2:
            fig, ax = plt.subplots(figsize=(10, 5))
            small_heights = df_display[df_display["height"] < 20]["height"]
            sns.histplot(small_heights, ax=ax, bins=20, kde=True)
            ax.set_title("Distribution of Small Bounding Box Heights (<20px)")
            ax.set_xlabel("Height (pixels)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            num_lt_5px_h = (df_display["height"] < 5).sum()
            st.metric(label="Boxes with Height < 5 pixels", value=f"{num_lt_5px_h:,}")
        # --- END NEW SECTION ---

    elif page == "Deep Dive by Category":
        st.header(f"Deep Dive by Category ({dataset_choice})")
        selected_category = st.sidebar.selectbox(
            "Select a Category to Analyze", sorted(df_display["category"].unique())
        )
        df_category = df_display[df_display["category"] == selected_category].copy()
        st.subheader(
            f"Analysis for: `{selected_category}` ({len(df_category):,} instances)"
        )

        st.markdown("#### Key Feature Distributions")
        col1, col2 = st.columns(2)
        with col1:
            plot_metric_distribution(
                df_category, "difficulty_score", "Distribution of Difficulty Scores"
            )
        with col2:
            plot_categorical_distribution(
                df_category, "location_context", "Distribution of Location Context"
            )

        st.markdown("---")
        st.markdown("#### Geometric & Size Properties")
        st.info(
            "Analysis of object shape, size, and position on the image plane (1280x720)."
        )
        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df_category["aspect_ratio"], kde=True, ax=ax, bins=50)
            ax.axvline(1.0, color="r", linestyle="--", label="Square (Ratio=1)")
            ax.set_title("Aspect Ratio Distribution", fontsize=16)
            ax.set_xlabel("Aspect Ratio (Width / Height)")
            ax.set_xlim(0, max(5, df_category["aspect_ratio"].quantile(0.98)))
            ax.legend()
            st.pyplot(fig)

        with col4:
            plot_log_transformed_distribution(
                df_category, "normalized_area", "Distribution of Normalized Area"
            )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist2d(
            df_category["centroid_x"],
            df_category["centroid_y"],
            bins=(50, 30),
            cmap="viridis",
        )
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)
        ax.invert_yaxis()
        ax.set_title("Object Centroid Heatmap", fontsize=16)
        ax.set_xlabel("Image X-coordinate")
        ax.set_ylabel("Image Y-coordinate")
        st.pyplot(fig)

    elif page == "Contextual Analysis":
        st.header(f"Contextual Analysis ({dataset_choice})")

        st.subheader("Scene vs. Class Density")
        st.info(
            "This chart shows the average number of objects of each class found per image within a given scene."
        )
        scene_class_counts = (
            df_display.groupby(["scene", "category"]).size().unstack(fill_value=0)
        )
        scene_image_counts = df_display.groupby("scene")["image_name"].nunique()
        avg_instances_per_image = scene_class_counts.div(scene_image_counts, axis=0)
        fig, ax = plt.subplots(figsize=(14, 7))
        avg_instances_per_image.plot(
            kind="bar", stacked=False, ax=ax, colormap="viridis"
        )
        ax.set_ylabel("Avg. Instances per Image")
        ax.set_xlabel("Scene")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("---")

        st.subheader("Interactive Environment vs. Object Visibility")
        st.info(
            "How object visibility attributes change with environment. Select a category to filter the analysis."
        )
        vis_cat_list = ["All Categories"] + sorted(df_display["category"].unique())
        vis_cat_selection = st.selectbox(
            "Select a Category for Visibility Analysis:", vis_cat_list
        )
        df_vis = (
            df_display.copy()
            if vis_cat_selection == "All Categories"
            else df_display[df_display["category"] == vis_cat_selection]
        )
        title_suffix = f"for '{vis_cat_selection}'"

        col1, col2 = st.columns(2)
        with col1:
            occlusion_by_time = (
                pd.crosstab(df_vis["timeofday"], df_vis["occluded"], normalize="index")
                * 100
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            occlusion_by_time.plot(kind="bar", stacked=True, ax=ax, colormap="coolwarm")
            ax.set_title(f"Occlusion Rate by Time of Day {title_suffix}")
            ax.set_ylabel("Percentage (%)")
            ax.tick_params(axis="x", rotation=0)
            st.pyplot(fig)
        with col2:
            truncation_by_time = (
                pd.crosstab(df_vis["timeofday"], df_vis["truncated"], normalize="index")
                * 100
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            truncation_by_time.plot(
                kind="bar", stacked=True, ax=ax, colormap="coolwarm"
            )
            ax.set_title(f"Truncation Rate by Time of Day {title_suffix}")
            ax.set_ylabel("Percentage (%)")
            ax.tick_params(axis="x", rotation=0)
            st.pyplot(fig)

        col3, col4 = st.columns(2)
        with col3:
            occlusion_by_weather = (
                pd.crosstab(df_vis["weather"], df_vis["occluded"], normalize="index")
                * 100
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            occlusion_by_weather.plot(
                kind="bar", stacked=True, ax=ax, colormap="coolwarm"
            )
            ax.set_title(f"Occlusion Rate by Weather {title_suffix}")
            ax.set_ylabel("Percentage (%)")
            ax.tick_params(axis="x", rotation=0)
            st.pyplot(fig)
        with col4:
            truncation_by_weather = (
                pd.crosstab(df_vis["weather"], df_vis["truncated"], normalize="index")
                * 100
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            truncation_by_weather.plot(
                kind="bar", stacked=True, ax=ax, colormap="coolwarm"
            )
            ax.set_title(f"Truncation Rate by Weather {title_suffix}")
            ax.set_ylabel("Percentage (%)")
            ax.tick_params(axis="x", rotation=0)
            st.pyplot(fig)

        st.markdown("---")

        st.subheader("Scene vs. Difficulty")
        st.info(
            "This analysis shows the distribution of difficulty scores across different scene types."
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_display, x="scene", y="difficulty_score", ax=ax)
        ax.set_title("Difficulty Score Distribution by Scene")
        ax.set_ylabel("Difficulty Score")
        ax.set_xlabel("Scene")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        st.markdown("---")

        st.subheader("Interactive Scene vs. Visibility Analysis")
        interactive_cat_scene = st.selectbox(
            "Select a Category for Scene Analysis:",
            sorted(df_display["category"].unique()),
        )
        df_scene_cat = df_display[df_display["category"] == interactive_cat_scene]
        col5, col6 = st.columns(2)
        with col5:
            occlusion_by_scene = (
                pd.crosstab(
                    df_scene_cat["scene"], df_scene_cat["occluded"], normalize="index"
                )
                * 100
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            occlusion_by_scene.plot(
                kind="bar", stacked=True, ax=ax, colormap="coolwarm"
            )
            ax.set_title(f"Occlusion Rate of '{interactive_cat_scene}' by Scene")
            ax.set_ylabel("Percentage (%)")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
        with col6:
            truncation_by_scene = (
                pd.crosstab(
                    df_scene_cat["scene"], df_scene_cat["truncated"], normalize="index"
                )
                * 100
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            truncation_by_scene.plot(
                kind="bar", stacked=True, ax=ax, colormap="coolwarm"
            )
            ax.set_title(f"Truncation Rate of '{interactive_cat_scene}' by Scene")
            ax.set_ylabel("Percentage (%)")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

        st.markdown("---")

        st.subheader("Interactive: Weather vs. Average Object Size")
        weather_class = st.selectbox(
            "Select a class for weather analysis:",
            sorted(df_display["category"].unique()),
        )
        weather_df = df_display[df_display["category"] == weather_class]
        avg_size_by_weather = (
            weather_df.groupby("weather")["normalized_area"]
            .mean()
            .sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        avg_size_by_weather.plot(kind="bar", ax=ax)
        ax.set_title(f"Average Normalized Area of '{weather_class}' by Weather")
        ax.set_ylabel("Average Normalized Area")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)

        st.markdown("---")

        st.subheader("Interactive Location Analysis")
        loc_cat = st.selectbox(
            "Select a Category for Location Analysis:",
            sorted(df_display["category"].unique()),
        )
        df_loc_cat = df_display[df_display["category"] == loc_cat]
        col7, col8 = st.columns(2)
        with col7:
            occlusion_by_loc = (
                pd.crosstab(
                    df_loc_cat["location_context"],
                    df_loc_cat["occluded"],
                    normalize="index",
                )
                * 100
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            occlusion_by_loc.plot(kind="bar", stacked=True, ax=ax, colormap="coolwarm")
            ax.set_title(f"Occlusion Rate of '{loc_cat}' by Location")
            ax.set_ylabel("Percentage (%)")
            ax.tick_params(axis="x", rotation=0)
            st.pyplot(fig)
        with col8:
            truncation_by_loc = (
                pd.crosstab(
                    df_loc_cat["location_context"],
                    df_loc_cat["truncated"],
                    normalize="index",
                )
                * 100
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            truncation_by_loc.plot(kind="bar", stacked=True, ax=ax, colormap="coolwarm")
            ax.set_title(f"Truncation Rate of '{loc_cat}' by Location")
            ax.set_ylabel("Percentage (%)")
            ax.tick_params(axis="x", rotation=0)
            st.pyplot(fig)

    elif page == "Qualitative Sample Explorer":
        st.header(f"Qualitative Sample Explorer ({dataset_choice})")
        st.sidebar.markdown("### Explorer Filters")

        # Restore all filters
        filter_category = st.sidebar.multiselect(
            "Filter by Category",
            sorted(df_display["category"].unique()),
            default=sorted(df_display["category"].unique()),
        )
        filter_difficulty = st.sidebar.slider("Minimum Difficulty Score", 0, 5, 0)
        filter_location = st.sidebar.multiselect(
            "Filter by Location Context",
            sorted(df_display["location_context"].unique()),
            default=sorted(df_display["location_context"].unique()),
        )
        filter_time = st.sidebar.multiselect(
            "Filter by Time of Day",
            sorted(df_display["timeofday"].unique()),
            default=sorted(df_display["timeofday"].unique()),
        )
        filter_weather = st.sidebar.multiselect(
            "Filter by Weather",
            sorted(df_display["weather"].unique()),
            default=sorted(df_display["weather"].unique()),
        )

        # Apply all filters
        filtered_df = df_display[
            (df_display["category"].isin(filter_category))
            & (df_display["difficulty_score"] >= filter_difficulty)
            & (df_display["location_context"].isin(filter_location))
            & (df_display["timeofday"].isin(filter_time))
            & (df_display["weather"].isin(filter_weather))
        ]

        st.info(
            f"Found **{filtered_df['image_name'].nunique():,}** unique images matching your criteria."
        )

        if not filtered_df.empty:
            image_names_to_sample = filtered_df["image_name"].unique()
            num_samples = st.slider(
                "Number of samples to display",
                1,
                min(10, len(image_names_to_sample)),
                3,
            )

            if st.button("Load New Random Samples"):
                pass

            selected_images = np.random.choice(
                image_names_to_sample, num_samples, replace=False
            )

            for img_name in selected_images:
                st.markdown(f"---")
                image_path = image_dir_display / img_name
                # Get all objects that match the filter criteria FOR THAT IMAGE
                boxes_to_draw = filtered_df[filtered_df["image_name"] == img_name]

                # Get general context from the first object found in that image
                attrs = df_display[df_display["image_name"] == img_name].iloc[0]
                st.subheader(f"Image: `{img_name}`")
                st.write(
                    f"**Context:** {attrs['weather']}, {attrs['scene']}, {attrs['timeofday']}"
                )

                image_with_boxes = draw_boxes_on_image(image_path, boxes_to_draw)
                if image_with_boxes is not None:
                    st.image(image_with_boxes, use_column_width=True)
        else:
            st.warning("No images match the current filter criteria.")

else:
    st.error(
        "Could not load the selected dataset. Please ensure the processed .parquet files exist."
    )
