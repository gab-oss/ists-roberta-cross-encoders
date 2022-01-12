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

for filename in os.listdir(config["model_save_path"]):
    if filename.startswith("model"):
        model_name = config["model_save_path"] + filename
        num_epochs = model_name.split('_')[-1]

        logger.info("Read test dataset")
        test_samples = []
        with open(config['test_data'], 'r', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                test_samples.append([row['chunk1'], row['chunk2']])

        model = CrossEncoder(model_name)
        predictions = model.predict(test_samples)


        pred_scores = []
        pred_cont_scores = []
        pred_cont_int_scores = []
        for p in predictions:
            pred_scores.append(p)
            pred_cont_scores.append(p * 5)
            pred_cont_int_scores.append(round(p * 5))
            

        result = config['results'] + 'score_epochs_' + str(num_epochs) + '.tsv'
        if os.path.exists(result):
            os.remove(os.path.join(result))

        with open(config['test_data'], 'r', encoding='utf8') as fIn, open(result, 'a+', encoding='utf8') as result:
            lines_count = 0
            for line in fIn.readlines():
                if (lines_count == 0):
                    res_line = line[:-1] + '\t' + 'pred_scores\t' + 'pred_cont_scores\t' + 'pred_cont_int_scores\n'
                    lines_count += 1
                    print(res_line)
                    result.write(res_line)
                else:
                    res_line = line[:-1] + '\t' + str(pred_scores[lines_count - 1]) + '\t' + str(pred_cont_scores[lines_count - 1]) + '\t' + str(pred_cont_int_scores[lines_count - 1]) + '\n'
                    lines_count += 1
                    print(res_line)
                    result.write(res_line)