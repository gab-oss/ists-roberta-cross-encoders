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
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    prev_sentence_id = -1
    for row in reader:
        sentence_id = row['sentence_id']
        chunks_ids = row['chunks_ids']
        type = row['type']
        score = row['score']
        chunk1 = row['chunk1']
        chunk2 = row['chunk2']
        pred_type = row['pred_types']
        pred_score = row['pred_scores']
        pred_cont_score = row['pred_cont_scores']
        pred_cont_int_score = row['pred_cont_int_scores']

        print(sentence_id)
        print(prev_sentence_id)
        if sentence_id != prev_sentence_id:
            if prev_sentence_id != -1:
                gs.write('</alignment>\n')
                gs.write('</sentence>\n')
            gs.write('<sentence id="{}" status="">\n'.format(sentence_id))
            gs.write('<alignment>\n')
        gs.write('{} // {} // {} // {} <==> {}\n'.format(chunks_ids, pred_type, pred_cont_int_score, chunk1, chunk2))
        prev_sentence_id = sentence_id

    gs.write('</alignment>\n')
    gs.write('</sentence>\n')
