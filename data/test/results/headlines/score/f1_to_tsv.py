import os
import argparse
import csv

parser = argparse.ArgumentParser(description='Read paths')
parser.add_argument('f1_path', type=str, help='path to f1 scores')
parser.add_argument('tsv_path', type=str, help='path to tsv output')

args = vars(parser.parse_args())

f1_path = args['f1_path']
tsv_path = args['tsv_path']

with open(f1_path, 'rt', encoding='utf8') as fIn, open(tsv_path, 'a+', encoding='utf8') as tsv:
    tsv.write('{}\t{}\t{}\t{}\t{}\n'.format('Epoch', 'Ali', 'Type', 'Score', 'Typ+Sco'))
    epoch = ''
    ali = ''
    typ = ''
    sco = ''
    ts = ''
    for line in fIn.readlines():
        if '.wa' in line:
            epoch = line.split('.')[0].split('_')[-1]
        elif 'Ali' in line:
            ali = line.split(' ')[-1][:-1]
        elif 'Type' in line:
            typ = line.split(' ')[-1][:-1]
        elif 'Score' in line:
            sco = line.split(' ')[-1][:-1]
        elif 'Typ+Sco' in line:
            ts = line.split(' ')[-1][:-1]
            print('{}\t{}\t{}\t{}\t{}\n'.format(epoch, ali, typ, sco, ts))
            tsv.write('{}\t{}\t{}\t{}\t{}\n'.format(epoch, ali, typ, sco, ts))


