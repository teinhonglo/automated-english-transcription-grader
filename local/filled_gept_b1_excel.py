import argparse
import random
import logging
import os
import csv
import numpy as np
from tqdm import tqdm
import pandas as pd

'''
產生111年口說語料.xlsx
'''

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_dir",
                    default="../corpus/speaking/GEPT_B1",
                    type=str)

parser.add_argument("--text_path",
                    default="../asr-esp/data/gept_b1/text",
                    type=str)

parser.add_argument("--recog_path",
                    default="../asr-esp/data/gept_b1/multi_en_mct_cnn_tdnnf_tgt3meg-dl/text",
                    type=str)

parser.add_argument("--scores",
                    default="content pronunciation vocabulary",
                    type=str)

parser.add_argument("--anno_fn",
                    default="111年口說語料.xlsx",
                    type=str)

parser.add_argument("--new_anno_fn",
                    default="new_111年口說語料.xlsx",
                    type=str)


args = parser.parse_args()

corpus_dir = args.corpus_dir
recog_path = args.recog_path
text_path = args.text_path
anno_path = os.path.join(corpus_dir, args.anno_fn)
scores = args.scores.split()
new_anno_path = os.path.join(corpus_dir, args.new_anno_fn)

text_dict = {}
recog_dict = {}

with open(text_path, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        uttid = info[0]
        content = " ".join(info[1:])
        if len(content) == 0: continue
        
        spkid, part, sub_qid, datatime = uttid.split("-")
        
        if spkid not in text_dict:
            text_dict[spkid] = {"2": ["" for _ in range(10)], "3": [""]}
        
        if part == "1":
            continue
        elif part == "2":
            text_dict[spkid]["2"][int(sub_qid) - 1] = content
        elif part == "3":
            text_dict[spkid]["3"][0] = content
        else:
            print("Something went wrong")
            exit(0)


with open(recog_path, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        uttid = info[0]
        content = " ".join(info[1:])
        if len(content) == 0: continue
        
        spkid, part, sub_qid, datatime = uttid.split("-")
        
        if spkid not in recog_dict:
            recog_dict[spkid] = {"2": ["" for _ in range(10)], "3": [""]}
        
        if part == "1":
            continue
        elif part == "2":
            recog_dict[spkid]["2"][int(sub_qid) - 1] = content
        elif part == "3":
            recog_dict[spkid]["3"][0] = content
        else:
            print("Something went wrong")
            exit(0)

anno_df_2nd = pd.read_excel(anno_path, sheet_name="2", converters={"編號名":str})
anno_df_2nd = anno_df_2nd[anno_df_2nd["編號名"].isin(list(recog_dict.keys()))].reset_index()
anno_df_3rd = pd.read_excel(anno_path, sheet_name="3", converters={"編號名":str})
anno_df_3rd = anno_df_3rd[anno_df_3rd["編號名"].isin(list(recog_dict.keys()))].reset_index()

print(anno_df_2nd.head())
for i in tqdm(range(len(anno_df_2nd))):
    spkid = anno_df_2nd["編號名"][i]
    text, recog_text = text_dict[spkid]["2"], recog_dict[spkid]["2"]
    
    for score in scores:
        # 四捨五入，因為只有兩個人評分，因此有小數點只會是.5
        # anno_df_2nd.at[i, score] = np.ceil(anno_df_2nd[score][i])
        anno_df_2nd.at[i, score] = anno_df_2nd[score][i]
    
    anno_df_2nd.at[i,"trans_human"] = " | ".join(text)
    anno_df_2nd.at[i, "trans_stt"] = " | ".join(recog_text)

print(anno_df_2nd.head())

print(anno_df_3rd.head())
for i in tqdm(range(len(anno_df_3rd))):
    spkid = anno_df_3rd["編號名"][i]
    text, recog_text = text_dict[spkid]["3"][0], recog_dict[spkid]["3"][0]
    
    for score in scores:
        # 四捨五入，因為只有兩個人評分，因此有小數點只會是.5
        #anno_df_3rd.at[i, score] = np.ceil(anno_df_3rd[score][i])
        anno_df_3rd.at[i, score] = anno_df_3rd[score][i]
    
    anno_df_3rd.at[i, "trans_human"] = text
    anno_df_3rd.at[i, "trans_stt"] = recog_text
print(anno_df_3rd.head())

# Write each dataframe to a different worksheet.
with pd.ExcelWriter(new_anno_path) as writer: 
    anno_df_2nd.to_excel(writer, sheet_name='2', index=False)
    anno_df_3rd.to_excel(writer, sheet_name='3', index=False)
