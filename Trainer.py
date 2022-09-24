import streamlit
from pipeline import run_pipeline


def start_training(dataset_name, dataset_subset_name, model_names):
    run_pipeline()
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
