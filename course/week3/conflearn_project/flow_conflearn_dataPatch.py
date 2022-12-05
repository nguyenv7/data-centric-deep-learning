"""This flow will train a neural network to perform sentiment classification 
for the beauty products reviews.
"""

import os
import torch
import random
import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader, TensorDataset

from metaflow import FlowSpec, step, Parameter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from cleanlab.filter import find_label_issues
from sklearn.model_selection import KFold

from src.system import ReviewDataModule, SentimentClassifierSystem
from src.utils import load_config, to_json, from_json
from src.consts import DATA_DIR


class TrainIdentifyReview(FlowSpec):
  r"""A MetaFlow that trains a sentiment classifier on reviews of luxury beauty
  products using PyTorch Lightning, identifies data quality issues using CleanLab, 
  and prepares them for review in LabelStudio.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default='./config.json')

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.next(self.init_system)

  @step
  def init_system(self):
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance.
    """
    # configuration files contain all hyperparameters
    config = load_config(self.config_path)

    # a data module wraps around training, dev, and test datasets
    dm = ReviewDataModule(config)

    # a PyTorch Lightning system wraps around model logic
    system = SentimentClassifierSystem(config)

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = config.train.ckpt_dir,
      monitor = 'dev_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    trainer = Trainer(
      max_epochs = config.train.optimizer.max_epochs,
      callbacks = [checkpoint_callback])

    # when we save these objects to a `step`, they will be available
    # for use in the next step, through not steps after.
    self.dm = dm
    self.system = system
    self.trainer = trainer
    self.config = config

    self.next(self.patch_annotation_data)

  @step
  def patch_annotation_data(self):  
    train_size = len(self.dm.train_dataset)
    dev_size = len(self.dm.dev_dataset)

    # merge all data
    self.all_df = pd.concat(
      [
        self.dm.train_dataset.data,
        self.dm.dev_dataset.data,
        self.dm.test_dataset.data
      ]
    )
    # patch data
    col_num = self.all_df.columns.get_loc('label')
    # load json pre annotation
    pre_ann = from_json(join(self.config.review.save_dir, 'pre-annotations.json'))
    for ann in pre_ann:
      ann_data = ann['predictions'][0]['result'][0]
      # get id
      sample_id = ann_data['id'].split(sep='-')[1]
      # get annotation
      sample_label = ann_data['value']['choices'][0]
      # patch
      self.all_df.iloc[int(sample_id), col_num] = 1 if sample_label == 'Positive' else 0
    
    # load json patch from Label Studio
    patch_ann = from_json(join(self.config.review.save_dir, 'pre-annotations-small-fixes.json'))
    for ann in patch_ann:
      ann_data = ann['annotations'][0]['result'][0]
      # get id
      sample_id = ann_data['id'].split(sep='-')[1]
      # get annotation
      sample_label = ann_data['value']['choices'][0]
      # patch
      self.all_df.iloc[int(sample_id), col_num] = 1 if sample_label == 'Positive' else 0
    
    self.next(self.train_test)
    
  @step
  def train_test(self):
    """Calls `fit` on the trainer.
    
    We first train and (offline) evaluate the model to see what 
    performance would be without any improvements to data quality.
    """
    dm = ReviewDataModule(self.config)
    train_size = len(dm.train_dataset)
    dev_size = len(dm.dev_dataset)
    
    dm.train_dataset.data = self.all_df.iloc[0:train_size]
    dm.dev_dataset.data = self.all_df.iloc[train_size:train_size+dev_size]
    dm.test_dataset.data = self.all_df.iloc[train_size+dev_size:]
    
    system = SentimentClassifierSystem(self.config)
    trainer = Trainer(
      max_epochs = self.config.train.optimizer.max_epochs)
    
    # Call `fit` on the trainer with `system` and `dm`.
    # Our solution is one line.
    trainer.fit(system, dm)
    trainer.test(system, dm, ckpt_path = 'best')

    # results are saved into the system
    results = system.test_results

    # print results to command line
    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs', 'dataPatch-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)
    
  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python flow_conflearn.py`. To list
  this flow, run `python flow_conflearn.py show`. To execute
  this flow, run `python flow_conflearn.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python flow_conflearn.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python flow_conflearn.py resume`
  
  You can specify a run id as well.
  """
  flow = TrainIdentifyReview()
