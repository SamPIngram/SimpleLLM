import streamlit as st
import os
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import simplellm
from simplellm.configurator import TrainerConfig

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

def train():
    trainer = simplellm.Trainer(config=st.session_state["config"])
    trainer.train()

st.sidebar.markdown("# Training Configuration")
st.sidebar.markdown("## Presets:")
presets()
st.sidebar.markdown("---")

st.sidebar.markdown("## Custom Configuration:")
out_dir = st.sidebar.text_input("Output Directory", "out")
eval_interval = st.sidebar.number_input("Evaluation Interval", 0, 999999, 2000)
log_interval = st.sidebar.number_input("Log Interval", 0, 999999, 1)
eval_iters = st.sidebar.number_input("Evaluation Iterations", 0, 999999, 200)
eval_only = st.sidebar.checkbox("Evaluation Only", False)
always_save_checkpoint = st.sidebar.checkbox("Always Save Checkpoint", True)
init_from = st.sidebar.selectbox("Initialize From", ["scratch", "resume", "gpt2"])

st.sidebar.markdown("## WandB Logging")
wandb_log = st.sidebar.checkbox("Enable WandB Logging", False)
wandb_project = st.sidebar.text_input("WandB Project", "SimpleLLM")
wandb_run_name = st.sidebar.text_input("WandB Run Name", "gpt2")

st.sidebar.markdown("## Data")
dataset = st.sidebar.selectbox("Dataset", ["openwebtext"])
gradient_accumulation_steps = st.sidebar.number_input("Gradient Accumulation Steps", 0, 999999, 5 * 8)
batch_size = st.sidebar.number_input("Batch Size", 0, 999999, 12)
block_size = st.sidebar.number_input("Block Size", 0, 999999, 1024)

st.sidebar.markdown("## Model")
n_layer = st.sidebar.number_input("Number of Layers", 0, 999999, 12)
n_head = st.sidebar.number_input("Number of Heads", 0, 999999, 12)
n_embd = st.sidebar.number_input("Number of Embeddings", 0, 999999, 768)
dropout = st.sidebar.number_input("Dropout", 0.0, 1.0, 0.0)
bias = st.sidebar.checkbox("Bias", False)

st.sidebar.markdown("## AdamW Optimizer")
learning_rate = st.sidebar.number_input("Learning Rate", 0.0, 1.0, 6e-4)
max_iters = st.sidebar.number_input("Maximum Iterations", 0, 999999, 600000)
weight_decay = st.sidebar.number_input("Weight Decay", 0.0, 1.0, 1e-1)
beta1 = st.sidebar.number_input("Beta 1", 0.0, 1.0, 0.9)
beta2 = st.sidebar.number_input("Beta 2", 0.0, 1.0, 0.95)
grad_clip = st.sidebar.number_input("Gradient Clip", 0.0, 1.0, 1.0)

st.sidebar.markdown("## Learning Rate Decay")
decay_lr = st.sidebar.checkbox("Decay Learning Rate", True)
warmup_iters = st.sidebar.number_input("Warmup Iterations", 0, 999999, 2000)
lr_decay_iters = st.sidebar.number_input("Learning Rate Decay Iterations", 0, 999999, 600000)
min_lr = st.sidebar.number_input("Minimum Learning Rate", 0.0, 1.0, 6e-5)

st.sidebar.markdown("## DDP Settings")
backend = st.sidebar.selectbox("Backend", ["nccl", "gloo"])

st.sidebar.markdown("---")


st.markdown("# Training")
st.button("Train", on_click=train)
