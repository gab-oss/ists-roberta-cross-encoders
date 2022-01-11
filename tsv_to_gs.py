import os
import argparse
import csv

parser = argparse.ArgumentParser(description='Read paths')
parser.add_argument('tsv_path', type=str, help='path to tsv input')
parser.add_argument('gs_path', type=str, help='path to save gold standard output')

args = vars(parser.parse_args())

tsv_file = args['tsv_path']
gs_path = args['gs_path']

with open(tsv_file, 'rt', encoding='utf8') as fIn, open(gs_path, 'a+', encoding='utf8') as gs:
    reader = fIn.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        prev_sentence_id = ''
        sentence_id = row['sentence_id']
        chunks_ids = row['chunks_ids']
        type = row['type']
        score = row['score']
        chunk1 = row['chunk1']
        chunk2 = row['chunk2']
        pred_types = row['pred_types']
        pred_scores = row['pred_scores']
        pred_cont_scores = row['pred_cont_scores']
        pred_cont_scores = row['pred_cont_scores']

        if sentence_id != prev_sentence_id:
            if prev_sentence_id != '':
                gs.write('</alignment>\n')
                gs.write('</sentence>\n')
            gs.write('<sentence id="{}" status="">\n'.format(sentence_id))
            gs.write('<alignment>\n')
            gs.write(' {} // {} // {} // {} <==> {}'.format(chunks_ids, type, score, chunk1, chunk2))

    gs.write('</alignment>\n')
    gs.write('</sentence>\n')
