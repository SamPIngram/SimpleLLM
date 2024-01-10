from transformers import AutoTokenizer

def huggingface(tokenizer_name):
    """Gets a tokenizer from HuggingFace."""
    return AutoTokenizer.from_pretrained(tokenizer_name)