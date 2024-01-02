import torch
import importlib

def auto_spec():
    # -----------------------------------------------------------------------------
    # automatically choose a device and dtype
    # -----------------------------------------------------------------------------
    if torch.cuda.is_available():
        device = 'cuda'
        compile = True
    elif torch.mps.is_available():
        device = 'mps'
        compile = False
    else:
        device = 'cpu'
        compile = False
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    return device, dtype, compile


class TrainerConfig:
    def __init__(self, config_fp=None):
        self.config_fp = config_fp
        for k,v in self.get().items():
            setattr(self, k, v)

    def __repr__(self):
        return str(self.get())

    def get(self):
        # -----------------------------------------------------------------------------
        # default config values designed to train a gpt2 (124M) on OpenWebText
        # I/O
        out_dir = 'out'
        eval_interval = 2000
        log_interval = 1
        eval_iters = 200
        eval_only = False # if True, script exits right after the first eval
        always_save_checkpoint = True # if True, always save a checkpoint after each eval
        init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
        # wandb logging
        wandb_log = False # disabled by default
        wandb_project = 'owt'
        wandb_run_name = 'gpt2' # 'run' + str(time.time())
        # data
        dataset = 'openwebtext'
        gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
        batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
        block_size = 1024
        # model
        n_layer = 12
        n_head = 12
        n_embd = 768
        dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
        bias = False # do we use bias inside LayerNorm and Linear layers?
        # adamw optimizer
        learning_rate = 6e-4 # max learning rate
        max_iters = 600000 # total number of training iterations
        weight_decay = 1e-1
        beta1 = 0.9
        beta2 = 0.95
        grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
        # learning rate decay settings
        decay_lr = True # whether to decay the learning rate
        warmup_iters = 2000 # how many steps to warm up for
        lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
        min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        # DDP settings
        backend = 'nccl' # 'nccl', 'gloo', etc.
        # system
        device, dtype, compile = auto_spec() # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
        # -----------------------------------------------------------------------------
        if self.config_fp is not None:
            spec = importlib.util.spec_from_file_location("user_config", self.config_fp)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        # -----------------------------------------------------------------------------
        config_keys = [k for k,v in locals().items() if not k.startswith('_') and k != "load" and isinstance(v, (int, float, bool, str))]
        config = {}
        for item in config_keys:
            if self.config_fp is not None and item in module.__dict__:
                config[item] = module.__dict__[item]
            else:
                config[item] = locals()[item]
        return config

if __name__ == '__main__':
    print(TrainerConfig(config_fp="configs/train_shakespeare.py"))