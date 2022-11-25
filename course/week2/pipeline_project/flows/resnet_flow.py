"""
This flow for testing resnet18 model performance on MNIST data
"""

import os
import torch
import random
import numpy as np
from os.path import join
from pathlib import Path
from pprint import pprint

from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.system import MNISTDataModule, DigitClassifierSystem
from src.tests.integration import MNISTIntegrationTest
from src.tests.regression import MNISTRegressionTest
from src.tests.directionality import MNISTDirectionalityTest
from src.utils import load_config, to_json


class DigitClassifierFlow(FlowSpec):
    flow_name = 'resnet18_flow'
    # global config path
    config_path = Parameter('config',
                            help='path to config file', default='./configs/resnet18_flow.json')

    def _getLogFile(self, json_name):
        """Make logfile and corresponding folder"""

        log_file = join(Path(__file__).resolve().parent.parent,
                        f'logs/{self.flow_name}', json_name)

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        return log_file

    @step
    def start(self):
        """Random seeds"""

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.next(self.init_system)

    @step
    def init_system(self):
        """Init data module, model module and Pytorch lightning trainer instance"""
        config = load_config(self.config_path)

        data_module = MNISTDataModule(config=config)
        system = DigitClassifierSystem(config=config)

        checkpoint_callback = ModelCheckpoint(
            dirpath=config.system.save_dir,
            monitor='dev_loss',
            mode='min',
            save_top_k=1,
            verbose=True,
        )

        trainer = Trainer(max_epochs=config.system.optimizer.max_epochs,
                          callbacks=[checkpoint_callback])

        self.data_module = data_module
        self.system = system
        self.trainer = trainer

        self.next(self.train_model)

    @step
    def train_model(self):
        """ Call trainer to fit the data with model"""
        self.trainer.fit(model=self.system, datamodule=self.data_module)

        self.next(self.offline_test)

    @step
    def offline_test(self):
        """Run default test"""
        self.trainer.test(self.system, self.data_module, ckpt_path='best')
        results = self.system.test_results

        pprint(results)

        log_file = self._getLogFile('offline-test-results.json')

        # save result
        to_json(results, log_file)

        self.next(self.integration_test)

    @step
    def integration_test(self):
        """Run integration test"""
        test = MNISTIntegrationTest()
        test.test(self.trainer, self.system)

        results = self.system.test_results
        pprint(results)
        log_file = self._getLogFile('integration-test-results.json')

        # save result
        to_json(results, log_file)
        
        self.next(self.regression_test)

    @step
    def regression_test(self):
        """Run regression test"""
        test = MNISTRegressionTest()
        test.test(self.trainer, self.system)

        results = self.system.test_results
        pprint(results)
        log_file = self._getLogFile('regression-test-results.json')

        # save result
        to_json(results, log_file)
        
        self.next(self.directionality_test)


    @step
    def directionality_test(self):
        """Run directionality test"""
        test = MNISTDirectionalityTest()
        test.test(self.trainer, self.system)

        results = self.system.test_results
        pprint(results)
        log_file = self._getLogFile('directionality-test-results.json')

        # save result
        to_json(results, log_file)
        
        self.next(self.end)


    @step
    def end(self):
        print('done!!!')


if __name__ == "__main__":
    flow = DigitClassifierFlow()
