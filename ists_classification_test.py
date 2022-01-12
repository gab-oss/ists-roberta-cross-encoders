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

        pred_types = []
        for p in predictions:
            pred_type = np.argmax(p)
            pred_types.append(list(label2int.keys())[list(label2int.values()).index(pred_type)])

        result = config['results'] + 'type_epochs_' + str(num_epochs) + '.tsv'
        if os.path.exists(result):
            os.remove(os.path.join(result))

        with open(config['test_data'], 'r', encoding='utf8') as fIn, open(result, 'a+', encoding='utf8') as result:
            lines_count = 0
            for line in fIn.readlines():
                if (lines_count == 0):
                    res_line = line[:-1] + '\t' + 'pred_types\n'
                    lines_count += 1
                    print(res_line)
                    result.write(res_line)
                else:
                    res_line = line[:-1] + '\t' + str(pred_types[lines_count - 1]) + '\n'
                    lines_count += 1
                    print(res_line)
                    result.write(res_line)




