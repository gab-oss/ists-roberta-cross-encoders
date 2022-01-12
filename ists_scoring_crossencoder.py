"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continious labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import os
import csv
from math import floor
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

model = CrossEncoder('roberta-base', num_labels=1)

# Read STSb dataset
logger.info("Read train dataset")

samples = []
with open(config["training_data"], 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = row['score']
        if score == 'NIL':
            score = 0
        score = float(score) / 5.0
        samples.append(InputExample(texts=[row['chunk1'], row['chunk2']], label=score))

split = 0.8
train_samples = samples[:floor(len(samples) * split)]
dev_samples = samples[floor(len(samples) * split):]

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=config["batch_size"])
evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='Scoring eval')

warmup_steps = math.ceil(len(train_dataloader) * config["number_of_epochs"] * config["warmup"]) 
logger.info("Warmup-steps: {}".format(warmup_steps))

def callback_save(score, epoch, steps):
    if epoch % 2 == 0:
        model.save(str(config["model_save_path"]) + 'model_score_epoch_' + str(epoch))
        print("Saved model to {}".format(str(config["model_save_path"]) + 'model_score_epoch_' + str(epoch)))

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=config["number_of_epochs"],
          warmup_steps=warmup_steps,
          output_path=config["model_save_path"],
          callback=callback_save)

