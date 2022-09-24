"""
Streamlit homepage to input dataset name and model name(s)
"""

import streamlit
from pipeline import run_pipeline


def start_training(dataset: str, dataset_subset: str, models: str) -> None:
    """Start training pipeline

    Args:
        dataset: Dataset to use for training
        dataset_subset: Subset of dataset to use for training
        models: Models to train

    Returns:
        None
    """

    if not dataset.strip():
        streamlit.error("Please enter a dataset name")
        return

    if not models:
        streamlit.error("Please select at least one model")
        return

    if not dataset_subset.strip():
        # Just to make sure it is not None
        dataset_subset = ""

    run_pipeline(dataset, dataset_subset, models)
    streamlit.balloons()
    streamlit.success("Training is just started")


streamlit.title("Auto NLP Classification")

streamlit.subheader("Dataset Selection")
dataset_name = streamlit.text_input("Dataset Name from Huggingface")
dataset_subset_name = streamlit.text_input("Name of the subset if any")

streamlit.subheader("Model Selection")
model_names = streamlit.text_input("Model name(s) from Huggingface")

streamlit.button(
    "Train",
    on_click=lambda: start_training(dataset_name, dataset_subset_name, model_names),
)
