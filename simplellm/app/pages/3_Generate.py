import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import simplellm
from simplellm.configurator import GeneratorConfig
import pandas as pd
import threading
import time
import glob

st.set_page_config(
    page_title="SimpleLLM",
    page_icon="🪄",
)

if "GeneratorConfig" not in st.session_state:
    st.session_state["GeneratorConfig"] = GeneratorConfig()
    if "config" in st.session_state:
        st.session_state["GeneratorConfig"].out_dir = st.session_state["config"].out_dir

def raw_string(string):
    return string.encode("unicode_escape").decode("utf-8")

def load_preset_config(config_fp):
    st.session_state["config"] = GeneratorConfig(config_fp=config_fp)
    print(f"Set config from: {config_fp}")

def presets():
    config_list = glob.glob("configs/*.py")
    for config in config_list:
        st.sidebar.button(os.path.basename(config), on_click=load_preset_config, args=(config,))

def update_config():
    st.session_state["GeneratorConfig"].device = device
    st.session_state["GeneratorConfig"].compile = compile
    st.session_state["GeneratorConfig"].dtype = dtype
    st.session_state["GeneratorConfig"].out_dir = out_dir
    st.session_state["GeneratorConfig"].start = start
    st.session_state["GeneratorConfig"].num_samples = num_samples
    st.session_state["GeneratorConfig"].max_new_tokens = max_new_tokens
    st.session_state["GeneratorConfig"].temperature = temperature
    st.session_state["GeneratorConfig"].top_k = top_k
    st.session_state["GeneratorConfig"].seed = seed
    st.session_state["GeneratorConfig"].to_file = to_file

def generate(session_state_config):
    simplellm.Generator(config=session_state_config).generate(pipe_stdout_to_gui=True)

st.markdown("# Generate Text")

st.sidebar.markdown("# Generate Configuration")

st.sidebar.markdown("## Presets:")
presets()
st.sidebar.markdown("---")
st.sidebar.markdown("## Custom Configuration:")
device = st.sidebar.selectbox("Device", ["cuda", "cpu", "mps"], ["cuda", "cpu", "mps"].index(st.session_state["config"].device))
compile = st.sidebar.checkbox("Compile", st.session_state["config"].compile)
dtype = st.sidebar.selectbox("Data Type", ["float16", "bfloat16"], ["float16", "bfloat16"].index(st.session_state["config"].dtype))
out_dir = st.sidebar.text_input("Model Directory", st.session_state["GeneratorConfig"].out_dir)
start = st.sidebar.text_input("Prompt", raw_string(st.session_state["GeneratorConfig"].start))
num_samples = st.sidebar.number_input("Number of Samples", 0, 999999, st.session_state["GeneratorConfig"].num_samples)
max_new_tokens = st.sidebar.number_input("Number of Tokens", 0, 999999, st.session_state["GeneratorConfig"].max_new_tokens)
temperature = st.sidebar.number_input("Temperature", 0.0, 1.0, st.session_state["GeneratorConfig"].temperature)
top_k = st.sidebar.number_input("Top K", 0, 999999, st.session_state["GeneratorConfig"].top_k)
seed = st.sidebar.number_input("Seed", 0, 999999, st.session_state["GeneratorConfig"].seed, help="random seed for reproducibility")
to_file = st.sidebar.text_input("File Name", st.session_state["GeneratorConfig"].to_file)
st.sidebar.button("Update Config", on_click=update_config)

with st.expander("Configuration"):
    config_table = pd.DataFrame.from_dict(st.session_state["GeneratorConfig"].__dict__, orient='index', columns=['Value'], dtype=str)
    st.table(config_table)

if st.button("Generate"):
    with open('stdout.txt', 'w') as file:
        file.write("Generating...\n")
    # Call your function in a separate thread and write its output to a file
    thread = threading.Thread(target=generate, args=(st.session_state["GeneratorConfig"],))
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