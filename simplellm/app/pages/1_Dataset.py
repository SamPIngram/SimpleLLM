import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import simplellm

st.set_page_config(
    page_title="SimpleLLM",
    page_icon="🪄",
)
config = "configs/shakespeare_config.py"

def get_shakespeare():
    return simplellm.get_input.tiny_shakespeare(config_fp=config)

st.sidebar.header("Dataset Configuration")
test_size = st.sidebar.slider("Test Size", 0.0, 1.0, 0.1)
seed = st.sidebar.number_input("Seed", 0, 999999, 110892)
shuffle = st.sidebar.checkbox("Shuffle", True)
dataset_key = st.sidebar.text_input("Dataset Key", "train")
num_proc = st.sidebar.number_input("Number of Processes", 1, os.cpu_count(), os.cpu_count())
tokenizer = st.sidebar.selectbox("Tokenizer", ["gpt2", "cl100k_base", "gpt-4"])

st.markdown("# Dataset")
st.button("Shakespeare", on_click=get_shakespeare)
st.button("OpenWebText")
