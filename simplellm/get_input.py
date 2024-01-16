import os
import numpy as np
import tiktoken
from datasets import Dataset, load_dataset # huggingface datasets
from simplellm.configurator import DataConfig
from tqdm import tqdm
import sys

def from_file(fp, config_fp=None, config=None, pipe_stdout_to_gui=False):
    """Gets the dataset from a file."""
    if config is None:
        config = DataConfig(config_fp=config_fp)
    with open(fp, 'r') as f:
            data = f.read()
    n = len(data)
    split = 1-config.test_size
    train_data = data[:int(n*split)]
    val_data = data[int(n*split):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding(config.tokenizer)
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    if pipe_stdout_to_gui:
        with open("stdout.txt", "a") as f:
            f.write(f"\ntrain has {len(train_ids):,} tokens")
            f.write(f"\nval has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), '..', 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), '..', 'val.bin'))

def tiny_shakespeare(config_fp=None):
    """Gets the Tiny Shakespeare dataset."""
    config = DataConfig(config_fp=config_fp)
    dataset = load_dataset("SamPIngram/tinyshakespeare")
    split_dataset = _split(dataset, config.test_size, config.seed, config.shuffle)
    tokenize = _tokenize(split_dataset, config.num_proc)
    _store(tokenize)

def openwebtext(config_fp=None, subset="all"):
    """Gets the OpenWebText dataset."""
    config = DataConfig(config_fp=config_fp)
    dataset = load_dataset("openwebtext", num_proc=config.num_proc)
    split_dataset = _split(dataset, config.test_size, config.seed, config.shuffle)
    tokenize = _tokenize(split_dataset, config.num_proc)
    _store(tokenize)

def huggingface_dataset(dataset_name, config_fp=None, config=None, pipe_stdout_to_gui=False):
    """Gets a HuggingFace dataset."""
    if config is None:
        config = DataConfig(config_fp=config_fp)
    original_stdout = sys.stdout
    with open('stdout.txt', 'a') as f: # doesn't work with st.write
        if pipe_stdout_to_gui:
            sys.stdout = f
        dataset = load_dataset(dataset_name, num_proc=config.num_proc)
        split_dataset = _split(dataset, config.test_size, config.seed, config.shuffle)
        tokenize = _tokenize(split_dataset, config.num_proc)
        _store(tokenize)
    sys.stdout = original_stdout

def _split(dataset: Dataset, test_size: float, seed: int, shuffle: bool):
    """Splits a dataset into train and test sets."""
    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed, shuffle=shuffle)
    split_dataset['val'] = split_dataset.pop('test') # rename test to val
    return split_dataset

def _process(data):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(data['text'])
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out

def _tokenize(dataset, num_proc):
    return dataset.map(
        _process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

def _store(tokenized):
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), '..', f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

if __name__ == "__main__":
    pass