import pandas as pd
import os
import argparse
import json
import re

"""
[音檔檔名規則]
u：userSN 學生編號
t ：testSN  場次編號
p ：paperSN  題本編號
i ：itemSN     題目編號
例如： u117_t9_p4_i16_1-2_20220922.wav
表示 學生117_場次9_題本9_題目16_題型一第二題_錄製日期.wav

分數範圍    等級
>=5   B2
[4.5, 5) B2B1
[4.0, 4.5) B1
[3.5, 4.0) B1A2
[3.0, 3.5) A2
[2.5, 3.0) A2A1
[2.0, 2.5) A1
<2  未達A1

output_csv_format

"""

parser = argparse.ArgumentParser()

parser.add_argument('--anno_path', type=str, default="/share/corpus/2023_teemi/annotation/口說預試試題評分資料-題本4v2.xlsx")
parser.add_argument('--local_corpus_dir', type=str, default="/share/nas167/teinhonglo/AcousticModel/spoken_test/corpus/speaking/teemi_pretest")
parser.add_argument('--question_types', type=str, default="基礎聽答,情境式提問與問答")
parser.add_argument('--data_dir', type=str, default="/share/nas167/teinhonglo/AcousticModel/spoken_test/asr-kaldi/data/pretest/2023_teemi")
parser.add_argument('--model_dir', type=str, default="multi_en_mct_cnn_tdnnf_tgt3meg-dl")

args = parser.parse_args()

all_question_types = ["基礎聽答", "情境式提問與問答", "主題式口說任務", "摘要報告", "計分說明"]
sub_question_dict = {"基礎聽答": "5", "情境式提問與問答": "3"}
question_dict = { qt: i + 1 for i, qt in enumerate(all_question_types)}

columns = ['test_book', 'test_book(sys)', 'student_id', 'content', 'pronunciation', 'vocabulary', 'weighted_score', 'wav_paths', 'trans_human', 'trans_stt']

anno_path = args.anno_path
data_dir = args.data_dir
model_dir = args.model_dir
local_corpus_dir = args.local_corpus_dir

wavscp_path = os.path.join(data_dir, "wav.scp")
recog_path = os.path.join(data_dir, model_dir, "text")
question_types = args.question_types.split(",")

wavscp_dict = {}
recog_dict = {}
anno_df_dict = {qt:pd.read_excel(anno_path, sheet_name=qt) for qt in all_question_types}

with open(wavscp_path, "r") as fn:
    for line in fn.readlines():
        wav_id, wav_path = line.split()
        wavscp_dict[wav_id] = wav_path

with open(recog_path, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_id = info[0]
        recog_dict[text_id] = " ".join(info[1:])
        
wav_path_text = "\n".join(list(wavscp_dict.keys()))

for qt in question_types:
    anno_df = anno_df_dict[qt]
    qt_id = question_dict[qt]
    sub_qnum = sub_question_dict[qt]
    
    for i in range(len(anno_df["test_book"])):
        tb_num = anno_df["test_book"][i]
        uid = anno_df["student_id"][i]
        # 依據題本決定資料夾
        dataset_dir = "tb{}".format(tb_num)
        result_dir = os.path.join(local_corpus_dir, dataset_dir)
        
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        
        try:
            wav_ids_list = anno_df["wav_paths"][i].split(",")
        except:
            continue
        
        wav_path_list = []
        recog_list = []
        
        for wav_id in wav_ids_list:
            wav_id = wav_id.split()[0]
            wav_path_list.append(wavscp_dict[wav_id])
            recog_list.append(recog_dict[wav_id])
        
        wav_paths = " | ".join(wav_path_list)
        recogs = " | ".join(recog_list)
        
        anno_df.at[i, "text_id"] = "tb{}_qt{}_u{}".format(tb_num, qt_id, uid)
        anno_df.at[i, "wav_paths"] = wav_paths
        anno_df.at[i, "trans_human"] = ""
        anno_df.at[i, "trans_stt"] = recogs
        
 
result_path = os.path.join(result_dir, "annotation_" + model_dir + ".xlsx")
with pd.ExcelWriter(result_path) as writer:    
    for qt in all_question_types:
        anno_df_dict[qt].to_excel(writer, sheet_name=qt, index=False)
