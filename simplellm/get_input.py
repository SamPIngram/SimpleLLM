import os
import numpy as np
import requests
import tiktoken

def tiny_shakespeare():
    """Returns the Tiny Shakespeare dataset."""
    return _get_text_dataset('tiny_shakespeare')

def openwebtext(subset="all"):
    """Returns the OpenWebText dataset."""
    return _get_text_dataset(f'openwebtext{subset}')

def _get_text_dataset(name):
    """Returns the dataset as a list of strings."""
    if name == 'tiny_shakespeare':
        # download the tiny shakespeare dataset
        input_file_path = os.path.join(os.path.dirname(__file__), '..', 'input.txt')
        if not os.path.exists(input_file_path):
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(input_file_path, 'w') as f:
                f.write(requests.get(data_url).text)

        with open(input_file_path, 'r') as f:
            data = f.read()
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        # encode with tiktoken gpt2 bpe
        enc = tiktoken.get_encoding("gpt2")
        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(os.path.join(os.path.dirname(__file__), '..', 'train.bin'))
        val_ids.tofile(os.path.join(os.path.dirname(__file__), '..', 'val.bin'))