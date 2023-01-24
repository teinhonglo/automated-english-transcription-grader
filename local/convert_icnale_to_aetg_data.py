import argparse
import random
import logging
import os
import csv
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import pandas as pd
'''
一個excel中含有train/dev/test
'''

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_dir",
                    default="../corpus/speaking/ICNALE",
                    type=str)

parser.add_argument("--data_dir",
                    default="data-speaking/icnale/train_stt_whisperv2_large",
                    type=str)

parser.add_argument("--anno_fn",
                    default="annotations_whisperv2_large.xlsx",
                    type=str)

parser.add_argument("--id_column",
                    default="id",
                    type=str)

parser.add_argument("--wav_column",
                    default="wav_path",
                    type=str)

parser.add_argument("--text_column",
                    default="trans_stt",
                    type=str)

parser.add_argument("--scores",
                    default="holistic",
                    type=str)

parser.add_argument("--kfold",
                    default="1",
                    type=int)

args = parser.parse_args()

corpus_dir = args.corpus_dir
data_dir = args.data_dir
anno_path = os.path.join(corpus_dir, args.anno_fn)
scores = args.scores.split()
id_column = args.id_column
text_column = args.text_column
wav_column = args.wav_column
kfold = args.kfold

partitions = ["train", "dev", "test"]
partitions_map = {"train": "train", "dev": "valid", "test": "test"}

xlsx_headers = ["text_id", "wav_path", "text" ] + scores

for nf in range(kfold):
    
    result_dir = os.path.join(data_dir, str(nf+1))
    
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    
    for part in partitions:
        anno_df = pd.read_excel(anno_path, sheet_name=part)
        tsv_dict = {h:[] for h in xlsx_headers}
        
        for i, text_id in tqdm(enumerate(anno_df[id_column])):
            text = anno_df[text_column][i]
            wav_path = anno_df[wav_column][i]

            if not isinstance(text_id, str) or not isinstance(text, str) or not isinstance(wav_path, str):
                print(i)
                continue
            elif len(text.split()) == 0:
                continue

            text = " ".join(text.split())
            tsv_dict["text_id"].append(text_id)
            tsv_dict["text"].append(text)
            tsv_dict["wav_path"].append(wav_path)
            
                
            for score in scores:
                anno_score = anno_df[score][i]
                tsv_dict[score].append(anno_score)
            
        tsv_df = pd.DataFrame.from_dict(tsv_dict)
        tsv_df.to_csv(os.path.join(result_dir, partitions_map[part] + ".tsv"), header=xlsx_headers, sep="\t", index=False)

