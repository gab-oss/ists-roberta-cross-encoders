"""

Usage:
python ists_classification_crossencoder.py

"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator, CESoftmaxAccuracyEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
import csv
from math import floor
import torch
import numpy as np
import argparse
import json 

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


parser = argparse.ArgumentParser(description='Read path to config')
parser.add_argument('config', type=str, help='path to config')
args = vars(parser.parse_args())

with open(args['config']) as conf:
    config = json.load(conf)

label2int = {"EQUI": 0, "REL": 1, "SIMI": 2, "SPE1": 3, "SPE2": 4, "OPPO": 5, "ALIC": 6, "NOALI": 7}

logger.info("Read train dataset")
samples = []
with open(config["training_data"], 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        samples.append(InputExample(texts=[row['chunk1'], row['chunk2']], label=label2int[row['type'].split('_')[0]]))

split = 0.8
train_samples = samples[:floor(len(samples) * split)]
dev_samples = samples[floor(len(samples) * split):]

model = CrossEncoder('roberta-base', num_labels=8)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=config["batch_size"])
evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples, name='Classification eval')

warmup_steps = math.ceil(len(train_dataloader) * config["number_of_epochs"] * config["warmup"])
logger.info("Warmup-steps: {}".format(warmup_steps))

def callback_save(score, epoch, steps):
    if epoch % 2 == 0:
        model.save(str(config["model_save_path"]) + 'model_type_epoch_' + str(epoch))
        print("Saved model to {}".format(str(config["model_save_path"]) + 'model_type_epoch_' + str(epoch)))

model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=config["number_of_epochs"],
          evaluation_steps=config["evaluation_steps"],
          warmup_steps=warmup_steps,
          output_path=config["model_save_path"],
          callback=callback_save)




