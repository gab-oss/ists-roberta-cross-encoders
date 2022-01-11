import os
import argparse

# python3 data/train/gs/STSint.gs.images.wa data/train/tsv/train_images.tsv data/train/cropped_gs/train_images_cropped.wa
# python3 data/train/gs/STSint.gs.headlines.wa data/train/tsv/train_headlines.tsv data/train/cropped_gs/train_headlines_cropped.wa

parser = argparse.ArgumentParser(description='Read paths')
parser.add_argument('gs_path', type=str, help='path to file with gold standard data')
parser.add_argument('tsv_path', type=str, help='path to save tsv output')
parser.add_argument('crop_path', type=str, help='path to save cropped output')

args = vars(parser.parse_args())

gs_path = args['gs_path']
tsv_file = args['tsv_path']
crop = args['crop_path']

if os.path.exists(tsv_file):
  os.remove(tsv_file)

if os.path.exists(crop):
  os.remove(crop)

with open(gs_path, 'rt', encoding='iso-8859-1') as fIn, open(crop, 'w', encoding='utf8') as crop, open(tsv_file, 'w', encoding='utf8') as fcsv:
    fcsv_head = "\t".join(['sentence_id', 'chunks_ids', 'type', 'score', 'chunk1', 'chunk2']) + '\n'
    fcsv.write(fcsv_head)

    lines_count = 0
    sentence_id = ''
    chunks_ids = ''
    type = ''
    score = ''
    chunk1 = ''
    chunk2 = ''
    for line in fIn.readlines():
        lines_count += 1
        if line.startswith('<sentence'):
            if (lines_count != 1):
                crop.write('</alignment>\n')
                crop.write('</sentence>\n\n')
            crop.write(line)
            crop.write('<alignment>\n')
            sentence_id = line.split('"')[1]
        elif '<==>' in line and '//' in line:
            alignment = line.split(' // ')
            chunks_ids = alignment[0]
            type = alignment[1]
            score = alignment[2]
            chunks = alignment[3].split(' <==> ')
            chunk1 = chunks[0]
            chunk2 = chunks[1][:-1] # remove space
            fcsv_line = "\t".join([sentence_id, chunks_ids, type, score, chunk1, chunk2]) + '\n'
            fcsv.write(fcsv_line)
            crop_line = chunks_ids + ' // ' + type + ' // ' + score + ' // ' + chunk1 + ' <==> ' + chunk2 + '\n'
            crop.write(crop_line)
        else:
            continue
    crop.write('</alignment>\n')
    crop.write('</sentence>\n\n')      

