import simplellm

# Get the Tiny Shakespeare dataset
simplellm.get_input.tiny_shakespeare()

# Train a model on the Tiny Shakespeare dataset
trainer = simplellm.Trainer(config_fp="configs/shakespeare_config.py")
trainer.train()

# Generate text from the trained model
simplellm.Generator(config_fp="configs/shakespeare_config.py").generate(to_file="generated.txt")