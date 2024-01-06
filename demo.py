import simplellm

simplellm.get_input.openwebtext()

trainer = simplellm.Trainer(config_fp="configs/train_shakespeare.py")

trainer.train()