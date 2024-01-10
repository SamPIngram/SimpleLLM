import os
import pickle
from contextlib import nullcontext
import torch
from simplellm.configurator import GeneratorConfig
from simplellm.models.transformer import TransformerConfig, Transformer
from transformers import AutoTokenizer

class Generator:
    def __init__(self, config_fp=None):
        self.config = GeneratorConfig(config_fp=config_fp)

    def generate(self):
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in self.config.device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config.dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # model
        if self.config.init_from == 'resume':
            # init from a model saved in a specific directory
            ckpt_path = os.path.join(self.config.out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.config.device)
            gptconf = TransformerConfig(**checkpoint['model_args'])
            model = Transformer(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
        elif self.config.init_from.startswith('gpt2'):
            # init from a given GPT-2 model
            model = Transformer.from_pretrained(self.config.init_from, dict(dropout=0.0))

        model.eval()
        model.to(self.config.device)
        if compile:
            model = torch.compile(model) # requires PyTorch 2.0 (optional)

        # look for the meta pickle in case it is available in the dataset folder
        load_meta = False
        if self.config.init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
            meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
            load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            # TODO want to make this more general to arbitrary encoder/decoder schemes
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        else:
            # ok let's assume gpt-2 encodings by default
            print("No meta.pkl found, assuming GPT-2 encodings...")
            enc = AutoTokenizer.from_pretrained("gpt2", add_special_tokens=True)
            encode = lambda s: enc.encode(s)
            decode = lambda l: enc.decode(l)

        # encode the beginning of the prompt
        if self.config.start.startswith('FILE:'):
            with open(self.config.start[5:], 'r', encoding='utf-8') as f:
                self.config.start = f.read()
        start_ids = encode(self.config.start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.config.device)[None, ...])

        # run generation
        if self.config.to_file is not False:
            with open(self.config.to_file, 'w', encoding='utf-8') as f:
                with torch.no_grad():
                    with ctx:
                        for k in range(self.config.num_samples):
                            y = model.generate(x, self.config.max_new_tokens, temperature=self.config.temperature, top_k=self.config.top_k)
                            f.write(decode(y[0].tolist()))
                            f.write('\n---------------\n')
        else:
            with torch.no_grad():
                with ctx:
                    for k in range(self.config.num_samples):
                        y = model.generate(x, self.config.max_new_tokens, temperature=self.config.temperature, top_k=self.config.top_k)
                        print(decode(y[0].tolist()))
                        print('---------------')