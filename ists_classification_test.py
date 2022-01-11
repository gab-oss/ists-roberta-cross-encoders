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


# train_path = 'data/train/'
# train_file = 'tsv/train_headlines.tsv'

test_path = 'data/test/'
test_file = 'tsv/test_headlines.tsv'

data_name = 'headlines'

model_save_path = 'model_type_epoch_16'

num_epochs = model_save_path.split('_')[-1]


label2int = {"EQUI": 0, "REL": 1, "SIMI": 2, "SPE1": 3, "SPE2": 4, "OPPO": 5, "ALIC": 6, "NOALI": 7}

# logger.info("Read train dataset")
# samples = []
# with open(os.path.join(train_path, train_file), 'r', encoding='utf8') as fIn:
#     reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#     for row in reader:
#         samples.append(InputExample(texts=[row['chunk1'], row['chunk2']], label=label2int[row['type'].split('_')[0]]))
#         #samples.append(InputExample(texts=[row['chunk2'], row['chunk1']], label=label2int[row['type']]))


# split = 0.8
# train_samples = samples[:floor(len(samples) * split)]
# dev_samples = samples[floor(len(samples) * split):]

logger.info("Read test dataset")
test_samples = []
with open(os.path.join(test_path, test_file), 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        #test_samples.append(InputExample(texts=[row['chunk1'], row['chunk2']], label=label2int[row['type']]))
        test_samples.append([row['chunk1'], row['chunk2']])

#Configuration
train_batch_size = 16

# output = 'models/' + data_name + '/type/'
# model_save_path = train_path + '_' + output

model = CrossEncoder(model_save_path)

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
# train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


# We add an evaluator, which evaluates the performance during training
# evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples, name='Classification eval')


# Configure the training
# warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
# logger.info("Warmup-steps: {}".format(warmup_steps))

# def callback_save(score, epoch, steps):
#     if epoch % 2 == 0:
#         model.save('model_type_epoch_' + str(epoch))
#         print("Saved model to {}".format('model_type_epoch_' + str(epoch)))

# # Train the model
# model.fit(train_dataloader=train_dataloader,
#           evaluator=evaluator,
#           epochs=num_epochs,
#           evaluation_steps=5000,
#           warmup_steps=warmup_steps,
#           output_path=model_save_path,
#           callback=callback_save)

predictions = model.predict(test_samples)
#print(predictions)

pred_types = []
for p in predictions:
    #print(p)
    pred_type = np.argmax(p)
    #print(pred_type)
    #print(list(label2int.keys())[list(label2int.values()).index(pred_type)])
    pred_types.append(list(label2int.keys())[list(label2int.values()).index(pred_type)])
    

result = 'results/headlines/type/' + 'type_epochs_' + str(num_epochs) + '.tsv'
if os.path.exists(os.path.join(test_path, result)):
  os.remove(os.path.join(test_path, result))

with open(os.path.join(test_path, test_file), 'r', encoding='utf8') as fIn, open(os.path.join(test_path, result), 'a+', encoding='utf8') as result:
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




