import argparse
import random
import logging
import os
import csv
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="feedback-prize-english-language-learning",
                    type=str)

parser.add_argument("--storage_data_dir",
                    default="data",
                    type=str)

parser.add_argument("--csv_fn",
                    default="train.csv",
                    type=str)

parser.add_argument("--is_train",
                    default="true",
                    type=str)

args = parser.parse_args()

def write_tsv(storage_data_dir, tsv_fn, tsv_dict, csv_header):
    tsv_path = os.path.join(storage_data_dir, tsv_fn)
    
    with open(tsv_path, 'w', newline='') as tsvfile:
        tsv_output = csv.writer(tsvfile, delimiter='\t')
        tsv_output.writerow(["text_id", "text", "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"])
        
        for text_id, scoring_info in tsv_dict.items():
            data = [text_id]
            for s in list(scoring_info.values()):
                data.append(s)
            
            tsv_output.writerow(data)


data_dir = args.data_dir
storage_data_dir = args.storage_data_dir
csv_path = os.path.join(data_dir, args.csv_fn)
is_train = args.is_train

tsv_dict = {}

with open(csv_path, 'r') as file:
    csv_reader = csv.reader(file, delimiter=",")
    
    for i, row in tqdm(enumerate(csv_reader)):
        if i != 0:
            if is_train == "true":
                text_id, text, cohesion, syntax, vocabulary, phraseology, grammar, conventions = row
            else:
                text_id, text = row
                cohesion, syntax, vocabulary, phraseology, grammar, conventions = 0, 0, 0, 0, 0, 0
            
            text = " ".join(text.split())
             
            tsv_dict[text_id] = {   
                                "text": text,
                                "cohesion": cohesion,
                                "syntax": syntax,
                                "vocabulary": vocabulary,
                                "phraseology": phraseology,
                                "grammar": grammar,
                                "conventions": conventions
                                }
        else:
            csv_header = row

num_egs = len(tsv_dict)

if is_train == "true":
    print("training/valid")
    num_egs_cv = len(tsv_dict) // 10
    num_egs_tr = num_egs - num_egs_cv
    tsv_list = list(tsv_dict.items())
    random.shuffle(tsv_list)
    tr_tsv_dict = dict(tsv_list[len(tsv_dict)//10:])
    cv_tsv_dict = dict(tsv_list[:len(tsv_dict)//10])  
    
    write_tsv(storage_data_dir, "train.tsv", tr_tsv_dict, csv_header)
    write_tsv(storage_data_dir, "valid.tsv", cv_tsv_dict, csv_header)
else:
    print("test")
    te_tsv_dict = dict(tsv_dict)
    write_tsv(storage_data_dir, "test.tsv", te_tsv_dict, csv_header)


