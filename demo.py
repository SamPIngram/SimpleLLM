import simplellm

# Use a config file to set all the parameters
config = "configs/shakespeare_config.py"

# Get the Tiny Shakespeare dataset
simplellm.get_input.tiny_shakespeare(config_fp=config)

# # Train a model on the Tiny Shakespeare dataset
# trainer = simplellm.Trainer(config_fp=config)
# trainer.train()

# Generate text from the trained model
simplellm.Generator(config_fp=config).generate()