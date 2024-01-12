##################################################
# Data config for Shakespeare
##################################################
test_size = 0.1
seed = 110892
shuffle = True
dataset_key = 'train'
num_proc = -1 # -1 for all, 1 for single process, 2 for two processes, etc.
tokenizer = 'gpt2' # 'gpt2' or 'cl100k_base' or 'gpt-4'

##################################################
# Training config for Shakespeare
##################################################
out_dir = '.outputs/out-shakespeare'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# Device settings (can enforce here or allow SimpleLLM to auto-detect)
# device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or 'mps' on macbooks
# dtype = 'bfloat16'
# compile = True # requires PyTorch 2.0 (optional), for mps this should be False

##################################################
# Generator config for Shakespeare
##################################################
# init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
# to_file = "generated.txt" # if not False, saves output to a file instead of printing to stdout