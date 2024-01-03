import simplellm

simplellm.get_input.tiny_shakespeare()

trainer = simplellm.Trainer(config_fp="configs/train_shakespeare.py")

trainer.train()