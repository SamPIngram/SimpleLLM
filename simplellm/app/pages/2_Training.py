import streamlit as st
import os
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import simplellm
from simplellm.configurator import TrainerConfig
import pandas as pd
import threading
import time

st.set_page_config(
    page_title="SimpleLLM",
    page_icon="🪄",
)

def presets():
    config_list = glob.glob("configs/*.py")
    for config in config_list:
        st.sidebar.button(os.path.basename(config), on_click=load_preset_config, args=(config,))

def load_preset_config(config_fp):
    st.session_state["config"] = TrainerConfig(config_fp=config_fp)
    print(f"Set config from: {config_fp}")

def update_config():
    st.session_state["config"].device = device
    st.session_state["config"].compile = compile
    st.session_state["config"].dtype = dtype
    st.session_state["config"].out_dir = out_dir
    st.session_state["config"].eval_interval = eval_interval
    st.session_state["config"].log_interval = log_interval
    st.session_state["config"].eval_iters = eval_iters
    st.session_state["config"].eval_only = eval_only
    st.session_state["config"].always_save_checkpoint = always_save_checkpoint
    st.session_state["config"].init_from = init_from
    st.session_state["config"].wandb_log = wandb_log
    st.session_state["config"].wandb_project = wandb_project
    st.session_state["config"].wandb_run_name = wandb_run_name
    st.session_state["config"].gradient_accumulation_steps = gradient_accumulation_steps
    st.session_state["config"].batch_size = batch_size
    st.session_state["config"].block_size = block_size
    st.session_state["config"].n_layer = n_layer
    st.session_state["config"].n_head = n_head
    st.session_state["config"].n_embd = n_embd
    st.session_state["config"].dropout = dropout
    st.session_state["config"].bias = bias
    st.session_state["config"].learning_rate = learning_rate
    st.session_state["config"].max_iters = max_iters
    st.session_state["config"].weight_decay = weight_decay
    st.session_state["config"].beta1 = beta1
    st.session_state["config"].beta2 = beta2
    st.session_state["config"].grad_clip = grad_clip
    st.session_state["config"].decay_lr = decay_lr
    st.session_state["config"].warmup_iters = warmup_iters
    st.session_state["config"].lr_decay_iters = lr_decay_iters
    st.session_state["config"].min_lr = min_lr
    st.session_state["config"].backend = backend

def train(session_state_config):
    trainer = simplellm.Trainer(config=session_state_config)
    trainer.train(pipe_stdout_to_gui=True)

if "config" not in st.session_state:
    st.session_state["config"] = TrainerConfig()

st.sidebar.markdown("# Training Configuration")
st.sidebar.markdown("## Presets:")
presets()
st.sidebar.markdown("---")

st.sidebar.markdown("## Custom Configuration:")
device = st.sidebar.selectbox("Device", ["cuda", "cpu", "mps"], ["cuda", "cpu", "mps"].index(st.session_state["config"].device))
compile = st.sidebar.checkbox("Compile", st.session_state["config"].compile)
dtype = st.sidebar.selectbox("Data Type", ["float16", "bfloat16"], ["float16", "bfloat16"].index(st.session_state["config"].dtype))
out_dir = st.sidebar.text_input("Output Directory", st.session_state["config"].out_dir)
eval_interval = st.sidebar.number_input("Evaluation Interval", 0, 999999, st.session_state["config"].eval_interval)
log_interval = st.sidebar.number_input("Log Interval", 0, 999999, st.session_state["config"].log_interval)
eval_iters = st.sidebar.number_input("Evaluation Iterations", 0, 999999, st.session_state["config"].eval_iters)
eval_only = st.sidebar.checkbox("Evaluation Only", st.session_state["config"].eval_only)
always_save_checkpoint = st.sidebar.checkbox("Always Save Checkpoint", st.session_state["config"].always_save_checkpoint)
init_from = st.sidebar.selectbox("Initialize From", ["scratch", "resume", "gpt2"], ["scratch", "resume", "gpt2"].index(st.session_state["config"].init_from))

st.sidebar.markdown("### WandB Logging")
wandb_log = st.sidebar.checkbox("Enable WandB Logging", st.session_state["config"].wandb_log)
wandb_project = st.sidebar.text_input("WandB Project", st.session_state["config"].wandb_project)
wandb_run_name = st.sidebar.text_input("WandB Run Name", st.session_state["config"].wandb_run_name)

st.sidebar.markdown("### Data")
gradient_accumulation_steps = st.sidebar.number_input("Gradient Accumulation Steps", 0, 999999, st.session_state["config"].gradient_accumulation_steps)
batch_size = st.sidebar.number_input("Batch Size", 0, 999999, st.session_state["config"].batch_size)
block_size = st.sidebar.number_input("Block Size", 0, 999999, st.session_state["config"].block_size)

st.sidebar.markdown("### Model")
n_layer = st.sidebar.number_input("Number of Layers", 0, 999999, st.session_state["config"].n_layer)
n_head = st.sidebar.number_input("Number of Heads", 0, 999999, st.session_state["config"].n_head)
n_embd = st.sidebar.number_input("Number of Embeddings", 0, 999999, st.session_state["config"].n_embd)
dropout = st.sidebar.number_input("Dropout", 0.0, 1.0, st.session_state["config"].dropout)
bias = st.sidebar.checkbox("Bias", st.session_state["config"].bias)

st.sidebar.markdown("### AdamW Optimizer")
learning_rate = st.sidebar.number_input("Learning Rate", 0.0, 1.0, format="%.4e", value=st.session_state["config"].learning_rate, step=1e-4)
max_iters = st.sidebar.number_input("Maximum Iterations", 0, 999999, st.session_state["config"].max_iters)
weight_decay = st.sidebar.number_input("Weight Decay", 0.0, 1.0, st.session_state["config"].weight_decay)
beta1 = st.sidebar.number_input("Beta 1", 0.0, 1.0, st.session_state["config"].beta1, step=0.01)
beta2 = st.sidebar.number_input("Beta 2", 0.0, 1.0, st.session_state["config"].beta2, step=0.01)
grad_clip = st.sidebar.number_input("Gradient Clip", 0.0, 1.0, st.session_state["config"].grad_clip, step=0.01)

st.sidebar.markdown("### Learning Rate Decay")
decay_lr = st.sidebar.checkbox("Decay Learning Rate", st.session_state["config"].decay_lr)
warmup_iters = st.sidebar.number_input("Warmup Iterations", 0, 999999, st.session_state["config"].warmup_iters)
lr_decay_iters = st.sidebar.number_input("Learning Rate Decay Iterations", 0, 999999, st.session_state["config"].lr_decay_iters)
min_lr = st.sidebar.number_input("Minimum Learning Rate", 0.0, 1.0, format="%.4e", value=st.session_state["config"].min_lr, step=1e-5)

st.sidebar.markdown("### DDP Settings")
backend = st.sidebar.selectbox("Backend", ["nccl", "gloo"], ["nccl", "gloo"].index(st.session_state["config"].backend))

st.sidebar.button("Update", on_click=update_config)
st.sidebar.markdown("---")


st.markdown("# Training")

with st.expander("Configuration"):
    config_table = pd.DataFrame.from_dict(st.session_state["config"].__dict__, orient='index', columns=['Value'], dtype=str)
    st.table(config_table)

if st.button("Train"):
    with open('stdout.txt', 'w') as file:
        file.write("Training...\n")
    # Call your function in a separate thread and write its output to a file
    thread = threading.Thread(target=train, args=(st.session_state["config"],))
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
