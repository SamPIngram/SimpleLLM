import unittest
import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simplellm import Trainer, Generator, get_input
from simplellm.configurator import TrainerConfig, GeneratorConfig, DataConfig

class TestSimpleLLM(unittest.TestCase):

    def setUp(self):
        self.trainer_config = TrainerConfig()
        self.generator_config = GeneratorConfig()
        self.data_config = DataConfig()
        self.trainer_config.out_dir = "test_out"
        self.generator_config.out_dir = "test_out"

    def test_trainer(self):
        trainer = Trainer(config=self.trainer_config)
        self.assertIsInstance(trainer, Trainer)

    def test_generator(self):
        generator = Generator(config=self.generator_config)
        self.assertIsInstance(generator, Generator)

    def test_get_input(self):
        get_input.huggingface_dataset("SamPIngram/tinyshakespeare",config=self.data_config)
        self.assertTrue(os.path.exists("train.bin"))
        self.assertTrue(os.path.exists("val.bin"))

    def test_trainer_train(self):
        self.trainer_config.max_iters=1
        self.trainer_config.eval_interval=1
        self.trainer_config.eval_iters=1
        self.trainer_config.warmup_iters=1
        self.trainer_config.n_layer=1
        self.trainer_config.n_head=1
        trainer = Trainer(config=self.trainer_config)
        try:
            trainer.train()
            exception_raised = False
        except:
            exception_raised = True
        self.assertFalse(exception_raised)

    def test_generator_generate(self):
        self.generator_config.num_samples=1
        self.generator_config.max_new_tokens=10
        generator = Generator(config=self.generator_config)
        try:
            generator.generate()
            exception_raised = False
        except:
            exception_raised = True
        self.assertFalse(exception_raised)

if __name__ == '__main__':
    unittest.main()
    shutil.rmtree("test_out/")