# SimpleLLM

## Preamble
SimpleLLM is a WIP basic implementation for training Large Language Models. This project will aim to include both a working module for training and generating text from created language models, as well as a series of tutorial notebooks for explaining each part of the process. The initial implmentation will be loosely based on the incredible work of the [NanoGPT project](https://github.com/karpathy/nanoGPT), but will be expanded to include more model architectures (e.g. selective  structured state space models) and a more modular design (multiple tokenizers and tools for running experiments).

During development I will attempt to leave [demo.py](https://github.com/SamPIngram/SimpleLLM/blob/main/demo.py) in a working state, so that you can run it to see the current state of the project. 

## Installation
SimpleLLM is currently only available on GitHub, so you will need to clone the repository to use it. You can do this by running the following command in your terminal:
```
git clone https://github.com/SamPIngram/SimpleLLM.git
```
Once you have cloned the repository, you can install the required packages by running the following command in the root directory of the project:
```
pip install -r requirements.txt
```

## Usage
SimpleLLM is designed to be as simple to use as possible. The following code snippet shows how to train a model on a text file and then generate text from it:
```python
# demo.py
import simplellm

# Use a config file to set all the parameters
config = "configs/shakespeare_config.py"

# Get the Tiny Shakespeare dataset
simplellm.get_input.tiny_shakespeare(config_fp=config)

# Train a model on the Tiny Shakespeare dataset
trainer = simplellm.Trainer(config_fp=config)
trainer.train()

# Generate text from the trained model
simplellm.Generator(config_fp=config).generate()
```

This can run by running the following command in the root directory of the project:
```bash
python demo.py
```
Training can also be done using muiltiple GPUs by running the following command (correcting the number of GPUs):
```bash
torchrun --standalone --nproc_per_node=8 demo.py
```
Or if you have a multiple cluster of GPUs, you can run the following command (correcting the number of nodes, node rank, master address and master port):
```bash
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 demo.py
```

## Roadmap
The following is a list of features that I would like to add to SimpleLLM in the future:
- [x] Add a basic implementation of a language model
- [x] Add a basic implementation of a trainer
- [x] Add a basic implementation of a generator
- [x] Add a basic implementation of a tokenizer
- [x] Add a basic implementation of getting datasets
- [ ] Add a GUI for running training experiments
- [ ] Add other model architectures:
    - [ ] selective structured state space models
    - [ ] mixture of experts models
- [ ] Add a tutorial notebook for each part of the process

## Contributing
If you would like to contribute to SimpleLLM, please feel free to open a pull request. If you have any questions about the project, please feel free to open an issue.

## License
SimpleLLM is licensed under the MIT license. See [LICENSE](https://github.com/SamPIngram/SimpleLLM/blob/main/LICENSE) for more details.
