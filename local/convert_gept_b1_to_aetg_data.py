import argparse
import random
import logging
import os
import csv
from sklearn.model_selection import train_test_split
import numpy as np
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

parser.add_argument("--sheet_name",
                    default="2",
                    type=str)

parser.add_argument("--kfold",
                    default=5,
                    type=int)

parser.add_argument("--test_on_valid",
                    action="store_true")

parser.add_argument("--merge_below_b1",
                    action="store_true")

parser.add_argument("--do_round",
                    action="store_true")

args = parser.parse_args()

corpus_dir = args.corpus_dir
data_dir = args.data_dir
anno_path = os.path.join(corpus_dir, args.anno_fn)
scores = args.scores.split()
id_column = args.id_column
text_column = args.text_column
kfold = args.kfold
sheet_name = args.sheet_name

xlsx_headers = ["text_id", "wav_path", "text" ] + scores
tsv_dict = {h:[] for h in xlsx_headers}
anno_df = pd.read_excel(anno_path, sheet_name=sheet_name, converters={id_column:str})


for i, text_id in tqdm(enumerate(anno_df[id_column])):
    
    text = anno_df[text_column][i]
    wav_path = anno_df["wav_path"][i]
    
    if not isinstance(text_id, str) or not isinstance(text, str):
        print(i)
    
    text = " ".join(text.split())
    tsv_dict["text_id"].append(text_id)
    tsv_dict["wav_path"].append(wav_path)
    tsv_dict["text"].append(text)
    
    for score in scores:
        anno_score = anno_df[score][i]
        
        if args.merge_below_b1:
            if anno_score <= 4:
                anno_score = 4
        
        # 因為只有兩人，因此有小數點時只有.5，取ceil即為四捨五入     
        if args.do_round:
            anno_score = np.ceil(anno_score)
            
        tsv_dict[score].append(anno_score)
    
    before_text_id = text_id

  
from sklearn.model_selection import KFold
kf = KFold(n_splits=kfold, random_state=66, shuffle=True)

tsv_df = pd.DataFrame.from_dict(tsv_dict)

if args.test_on_valid:
    all_train_df = tsv_df
else:
    all_train_df, test_df = train_test_split(tsv_df, test_size=0.2, random_state=66)

for i, (train_index, valid_index) in enumerate(kf.split(all_train_df)):
    kfold_dir = str(i+1)
    result_dir = os.path.join(data_dir, kfold_dir)
    
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    
    train_df, valid_df = all_train_df.iloc[train_index], all_train_df.iloc[valid_index]
    
    train_df.to_csv(os.path.join(result_dir, "train.tsv"), header=xlsx_headers, sep="\t", index=False)
    valid_df.to_csv(os.path.join(result_dir, "valid.tsv"), header=xlsx_headers, sep="\t", index=False)
    
    if args.test_on_valid:
        test_df = valid_df
     
    test_df.to_csv(os.path.join(result_dir, "test.tsv"), header=xlsx_headers, sep="\t", index=False)

