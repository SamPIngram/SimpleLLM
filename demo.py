import simplellm

simplellm.get_input.tiny_shakespeare()

simplellm.trainer.Trainer(config_fp="configs/train_shakespeare.py").train()