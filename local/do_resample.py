import argparse
import random
import logging
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="data-speaking-rs-vocabulary",
                    type=str)

parser.add_argument("--score",
                    default="vocabulary",
                    type=str)
 
parser.add_argument("--n_resamples",
                    default="120",
                    type=int)

parser.add_argument("--tsv_fn",
                    default="train.tsv",
                    type=str)

args = parser.parse_args()

def do_resample(df, score="vocabulary", n_resamples=140, scales=[1,2,3,4,5,6,7,8], resample_scales=[1,2,4,6,8]):
    df_balanced = None

    for g in scales:
        sub_df = df.loc[df[score] == g]
        
        if len(sub_df) == 0: continue
 
        if g in resample_scales:
            df_copied = None
            n_rs = n_resamples
            while n_rs - len(sub_df) > 0:
                if df_copied is None:
                    df_copied = sub_df.copy()
                else:
                    df_copied = pd.concat([df_copied, sub_df.copy()], axis=0)
                n_rs -= len(sub_df)

            replace=True
            if len(sub_df) >= n_rs:
                replace=False
             
            df_resampled = resample(sub_df,
                                replace=replace,
                                n_samples=n_rs,
                                random_state=66)
            
            if df_copied is not None:
                df_resampled = pd.concat([df_copied, df_resampled], axis=0)
        else:
            df_resampled = sub_df.copy()
        
        print(g, len(sub_df), len(df_resampled), n_resamples)
        if df_balanced is None:
            df_balanced = df_resampled.copy()
        else:
            df_balanced = pd.concat([df_balanced, df_resampled], axis=0)
    
    return df_balanced

data_dir = args.data_dir
score = args.score
n_resamples = args.n_resamples
tsv_fn = args.tsv_fn
resample_scales = [1,2,4,6,8]

tsv_path = os.path.join(data_dir, tsv_fn)
df = pd.read_csv(tsv_path, sep='\t', dtype={"text_id":str})
df_balanced = do_resample(df=df, score=score, n_resamples=n_resamples, resample_scales=resample_scales)
df_balanced.to_csv(tsv_path, sep="\t", index=False)
#df_balanced.to_csv("train.tsv", sep="\t", index=False)

