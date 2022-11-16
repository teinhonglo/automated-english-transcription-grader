import argparse
import random
import logging
import os
import csv
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_dir",
                    default="../corpus/writing/GSAT",
                    type=str)

parser.add_argument("--data_dir",
                    default="data",
                    type=str)

parser.add_argument("--anno_fn",
                    default="109年寫作語料.xlsx",
                    type=str)

parser.add_argument("--id_column",
                    default="編號名",
                    type=str)

parser.add_argument("--text_column",
                    default="作文(工讀生校正)",
                    type=str)

parser.add_argument("--scores",
                    default="content organization grammar vocabulary",
                    type=str)

parser.add_argument("--kfold",
                    default=5,
                    type=int)

args = parser.parse_args()

corpus_dir = args.corpus_dir
data_dir = args.data_dir
anno_path = os.path.join(corpus_dir, args.anno_fn)
scores = args.scores.split()
id_column = args.id_column
text_column = args.text_column
kfold = args.kfold

xlsx_headers = ["text_id", "text" ] + scores
tsv_dict = {h:[] for h in xlsx_headers}
anno_df = pd.read_excel(anno_path)

for i, text_id in tqdm(enumerate(anno_df[id_column])):
    
    if not isinstance(text_id, str):
        print(i)
    
    text = anno_df[text_column][i]
    text = " ".join(text.split())
    tsv_dict["text_id"].append(text_id)
    tsv_dict["text"].append(text)
    
    for score in scores:
        tsv_dict[score].append(anno_df[score][i])
    
    before_text_id = text_id

  
from sklearn.model_selection import KFold
kf = KFold(n_splits=kfold, random_state=66, shuffle=True)

tsv_df = pd.DataFrame.from_dict(tsv_dict)

for i, (train_index, test_index) in enumerate(kf.split(tsv_df)):
    kfold_dir = str(i+1)
    result_dir = os.path.join(data_dir, kfold_dir)
    
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    
    train_df, test_df = tsv_df.iloc[train_index], tsv_df.iloc[test_index] 
    
    train_df.to_csv(os.path.join(result_dir, "train.tsv"), header=xlsx_headers, sep="\t", index=False)
    test_df.to_csv(os.path.join(result_dir, "valid.tsv"), header=xlsx_headers, sep="\t", index=False)
    test_df.to_csv(os.path.join(result_dir, "test.tsv"), header=xlsx_headers, sep="\t", index=False)    
