import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import simplellm
from simplellm.configurator import DataConfig
import glob
import pandas as pd
import threading
import time

st.set_page_config(
    page_title="SimpleLLM",
    page_icon="🪄",
)

if "DataConfig" not in st.session_state:
    st.session_state["DataConfig"] = DataConfig()

def get(session_state_config):
    if data_source == "File":
        simplellm.get_input.from_file(session_state_config.dataset_url, config=session_state_config, pipe_stdout_to_gui=True)
    elif data_source == "HuggingFace":
        simplellm.get_input.huggingface_dataset(session_state_config.dataset_url, config=session_state_config, pipe_stdout_to_gui=True)

def presets():
    config_list = glob.glob("configs/*.py")
    for config in config_list:
        st.sidebar.button(os.path.basename(config), on_click=load_preset_config, args=(config,))

def load_preset_config(config_fp):
    st.session_state["DataConfig"] = DataConfig(config_fp=config_fp)
    print(f"Set config from: {config_fp}")

def update_config():
    st.session_state["DataConfig"].test_size = test_size
    st.session_state["DataConfig"].seed = seed
    st.session_state["DataConfig"].shuffle = shuffle
    st.session_state["DataConfig"].dataset_key = dataset_key
    st.session_state["DataConfig"].num_proc = num_proc
    st.session_state["DataConfig"].tokenizer = tokenizer

st.sidebar.markdown("# Dataset Configuration")
st.sidebar.markdown("## Presets:")
presets()
st.sidebar.markdown("---")
st.sidebar.markdown("## Custom Configuration:")
test_size = st.sidebar.slider("Test Size", 0.0, 1.0, st.session_state["DataConfig"].test_size, 0.01)
seed = st.sidebar.number_input("Seed", 0, 999999, st.session_state["DataConfig"].seed)
shuffle = st.sidebar.checkbox("Shuffle", st.session_state["DataConfig"].shuffle)
dataset_key = st.sidebar.text_input("Dataset Key", st.session_state["DataConfig"].dataset_key)
num_proc = st.sidebar.number_input("Number of Processes", 0, os.cpu_count(), st.session_state["DataConfig"].num_proc)
tokenizer = st.sidebar.selectbox("Tokenizer", ["gpt2", "cl100k_base", "gpt-4"], ["gpt2", "cl100k_base", "gpt-4"].index(st.session_state["DataConfig"].tokenizer))
st.sidebar.button("Update Config", on_click=update_config)

st.markdown("# Dataset")
with st.expander("Configuration"):
    config_table = pd.DataFrame.from_dict(st.session_state["DataConfig"].__dict__, orient='index', columns=['Value'], dtype=str)
    st.table(config_table)
data_source = st.radio("Data Source:", ("HuggingFace", "File"))
st.session_state["DataConfig"].dataset_url = st.text_input("Dataset Path", st.session_state["DataConfig"].dataset_url)

if st.button("Download & Prepare Dataset"):
    with open('stdout.txt', 'w') as file:
        file.write("Downloading...\n")
    # Call your function in a separate thread and write its output to a file
    thread = threading.Thread(target=get, args=(st.session_state["DataConfig"],))
    thread.start()

    # Initialize the position to 0
    position = 0

    # Display the contents of the file in real-time
    while thread.is_alive():
        with open('stdout.txt', 'r') as file:
            # Seek to the position where we last stopped reading
            file.seek(position)
            # Read the new additions to the file
            new_additions = file.read()
            # Update the position
            position = file.tell()
        # Display the new additions
        if new_additions:
            st.text(new_additions)
        time.sleep(1)
    st.text("Done!")
