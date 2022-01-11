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
import sys
import os
import gzip
import csv
from math import floor

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout

# parser = argparse.ArgumentParser(description='Read paths and number of epochs')

# parser.add_argument('train_path', type=str, help='path to file with tsv training file and to save models')
# parser.add_argument('train_file', type=str, help='name of tsv training file')

# parser.add_argument('test_path', type=str, help='path with test tsv and to save tsv output')
# parser.add_argument('test_file', type=str, help='name of test file')

# parser.add_argument('data_name', type=str, help='convinient name of dataset')

# parser.add_argument('num_epochs', type=str, help='number of epochs')

# args = vars(parser.parse_args())

# train_path = args['train_path']
# train_file = args['train_file']

# test_path = args['test_path']
# test_file = args['test_file']

# data_name = args['data_name']

# num_epochs = args['num_epochs']


# train_path = 'data/train/tsv/'
# train_file = 'train_headlines.tsv'

test_path = 'data/test/results/headlines/type/'
test_file = 'type_epochs_0.tsv'

data_name = 'headlines'

model_save_path = 'model_score_epoch_2'

num_epochs = model_save_path.split('_')[-1]

#We use distilroberta-base as base model and set num_labels=1, which predicts a continous score between 0 and 1
model = CrossEncoder(model_save_path)

logger.info("Read test dataset")
test_samples = []
with open(os.path.join(test_path, test_file), 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        #test_samples.append(InputExample(texts=[row['chunk1'], row['chunk2']], label=label2int[row['score']]))
        test_samples.append([row['chunk1'], row['chunk2']])

#Configuration
train_batch_size = 16

##### Load model and eval on test set
model = CrossEncoder(model_save_path)

# evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name='sts-test')
# evaluator(model)


predictions = model.predict(test_samples)
print(predictions)

pred_scores = []
pred_cont_scores = []
pred_cont_int_scores = []
for p in predictions:
    print(p)
    pred_scores.append(p)
    pred_cont_scores.append(p * 5)
    pred_cont_int_scores.append(round(p * 5))
    

result = 'results/headlines/score/' + 'score_epochs_' + str(num_epochs) + '.tsv'
if os.path.exists(os.path.join(test_path, result)):
  os.remove(os.path.join(test_path, result))

with open(os.path.join(test_path, test_file), 'r', encoding='utf8') as fIn, open(os.path.join('data/test/', result), 'a+', encoding='utf8') as result:
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