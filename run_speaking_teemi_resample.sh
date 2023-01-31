
hostname=`hostname`
stage=0
stop_stage=1000
# data-related
corpus_dir="../corpus/speaking/teemi_pretest"
score_names="content pronunciation vocabulary"
anno_fn="annotation_multi_en_mct_cnn_tdnnf_tgt3meg-dl.xlsx"
kfold=5
test_on_valid="true"
merge_below_b1="false"
trans_type="trans_stt"
do_round="true"
# resample-related
n_resamples=max
origin_scales="1,2,3,4,5,6,7,8"
resample_scales="1,2,3,4,5,6,7,8"
# model-related
model=pool
exp_tag=bert-pool-model
model_path=bert-base-uncased
max_score=5
max_seq_length=128
num_epochs=6
score_loss="mse"
# training-related
warmup_steps=0  # 0
weight_decay=0  # 0
max_grad_norm=1.0   # 1.0
train_batch_size=8  # 8
gradient_accumulation_steps=1  # 1
num_epochs=6        # 6
learning_rate=5e-5  # 5e-5
# other
rprefix=
score_loss=mse
test_book=1
part=1 # 1 = 基礎聽答, 2 = 情境式提問與問答, 3 = 主題式口說任務, 4 = 摘要報告 (不自動評分) 
do_split=true
do_dig=true
ori_all_bins="1,2,2.5,3,3.5,4,4.5,5" # 1 和 1.5 當作同一類
all_bins="1.5,2.5,3.5,4.5,5.5,6.5,7.5"
cefr_bins="1.5,3.5,5.5,7.5"
extra_options=

. ./path.sh
. ./parse_options.sh

set -euo pipefail

folds=`seq 1 $kfold`
corpus_dir=${corpus_dir}/tb${test_book}

data_dir=data-speaking/teemi-tb${test_book}p${part}/${trans_type}
exp_root=exp-speaking/teemi-tb${test_book}p${part}/${trans_type}
runs_root=runs-speaking/teemi-tb${test_book}p${part}/${trans_type}

if [ "$test_on_valid" == "true" ]; then
    extra_options="$extra_options --test_on_valid"
    data_dir=${data_dir}_tov
    exp_root=${exp_root}_tov
    runs_root=${runs_root}_tov
fi

if [ "$do_dig" == "true" ]; then
    # [0, 1, 1.5, 2, 2.78, 3.5, 4, 4.25, 5, 4.75] -> [0, 1, 2, 3, 4, 6, 7, 7, 9, 8]
    extra_options="$extra_options --do_dig"
else
    data_dir=${data_dir}_wod
    exp_root=${exp_root}_wod
    runs_root=${runs_root}_wod
fi

if [ "$do_split" == "true" ]; then
    # 一個音檔當一個
    extra_options="$extra_options --do_split"
else
    data_dir=${data_dir}_nosp
    exp_root=${exp_root}_nosp
    runs_root=${runs_root}_nosp
fi


if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then  
    if [ -d $data_dir ]; then
        echo "[NOTICE] $data_dir is already existed."
        echo "Skip data preparation."
        sleep 5
    else
        python local/convert_teemi_to_aetg_datav2.py \
                    --corpus_dir $corpus_dir \
                    --data_dir $data_dir \
                    --anno_fn $anno_fn \
                    --id_column "text_id" \
                    --text_column "$trans_type" \
                    --scores "$score_names" \
                    --sheet_name $part \
                    --all_bins $ori_all_bins \
                    --kfold $kfold $extra_options
    fi
fi


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for sn in $score_names; do
        new_data_dir=${data_dir}_${rprefix}r${n_resamples}_${sn}
        new_exp_root=${exp_root}_${rprefix}r${n_resamples}_${sn}
        new_runs_root=${runs_root}_${rprefix}r${n_resamples}_${sn}
        
        echo $new_data_dir
        if [ ! -d $new_data_dir ]; then
            mkdir -p $new_data_dir;
            rsync -a --exclude=*cached* $data_dir/ $new_data_dir/
            for fd in $folds; do
                python local/do_resample.py --data_dir $new_data_dir/$fd \
                                        --score $sn \
                                        --origin_scales $origin_scales \
                                        --resample_scales $resample_scales \
                                        --n_resamples $n_resamples
            done
        else
            echo "$new_data_dir is already existed"
        fi
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
     
    for sn in $score_names; do
        for fd in $folds; do
            # model_args_dir
            new_data_dir=${data_dir}_${rprefix}r${n_resamples}_${sn}
            new_exp_root=${exp_root}_${rprefix}r${n_resamples}_${sn}
            new_runs_root=${runs_root}_${rprefix}r${n_resamples}_${sn}
            output_dir=$exp_tag/${sn}/${fd}
            model_args_dir=$exp_tag/${sn}/${fd}

            if [ -d $new_exp_root/$model_args_dir/final ]; then
                echo "$new_exp_root/$model_args_dir/final is already existed."
                continue
            fi
            
            python3 run_speech_grader.py --do_train --save_best_on_evaluate --save_best_on_train \
                                         --do_lower_case --overwrite_cache \
                                         --model $model \
                                         --model_path $model_path \
                                         --num_train_epochs $num_epochs \
                                         --weight_decay $weight_decay \
                                         --max_grad_norm $max_grad_norm \
                                         --logging_steps 20 \
                                         --train_batch_size $train_batch_size \
                                         --gradient_accumulation_steps $gradient_accumulation_steps \
                                         --warmup_steps $warmup_steps \
                                         --learning_rate $learning_rate \
                                         --max_seq_length $max_seq_length \
                                         --max_score $max_score --evaluate_during_training \
                                         --score_loss $score_loss \
                                         --output_dir $output_dir \
                                         --score_name $sn \
                                         --data_dir $new_data_dir/$fd \
                                         --runs_root $new_runs_root \
                                         --exp_root $new_exp_root
        done
    done
fi



if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then  
    
    for sn in $score_names; do
        for fd in $folds; do
            new_data_dir=${data_dir}_${rprefix}r${n_resamples}_${sn}
            new_exp_root=${exp_root}_${rprefix}r${n_resamples}_${sn}
            new_runs_root=${runs_root}_${rprefix}r${n_resamples}_${sn}
            
            output_dir=$exp_tag/${sn}/${fd}
            model_args_dir=$exp_tag/${sn}/${fd}
            model_dir=$model_args_dir/best
            predictions_file="$new_runs_root/$output_dir/predictions.txt"
            
            python3 run_speech_grader.py --do_test --model $model \
                                         --do_lower_case \
                                         --model_path $model_path \
                                         --model_args_dir $model_args_dir \
                                         --max_seq_length $max_seq_length \
                                         --max_score $max_score \
                                         --model_dir $model_dir \
                                         --predictions_file $predictions_file \
                                         --data_dir $new_data_dir/$fd \
                                         --score_name $sn \
                                         --runs_root $new_runs_root \
                                         --output_dir $output_dir \
                                         --exp_root $new_exp_root
        done
    done 
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then  
    for sn in $score_names; do
        new_data_dir=${data_dir}_${rprefix}r${n_resamples}_${sn}
        new_runs_root=${runs_root}_${rprefix}r${n_resamples}_${sn}
        python local/speaking_predictions_to_report.py  --data_dir $new_data_dir \
                                                    --result_root $new_runs_root/$exp_tag \
                                                    --all_bins "$all_bins" \
                                                    --cefr_bins "$cefr_bins" \
                                                    --folds "$folds" \
                                                    --scores "$score_names" > $new_runs_root/$exp_tag/report.log
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then  
    echo $runs_root/$exp_tag
    for sn in $score_names; do
        new_data_dir=${data_dir}_${rprefix}r${n_resamples}_${sn}
        new_runs_root=${runs_root}_${rprefix}r${n_resamples}_${sn}
        python local/visualization.py   --result_root $new_runs_root/$exp_tag \
                                        --all_bins "$all_bins" \
                                        --cefr_bins "$cefr_bins" \
                                        --scores "$sn"
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then  
    for sn in $score_names; do
        new_data_dir=${data_dir}_${rprefix}r${n_resamples}_${sn}
        new_runs_root=${runs_root}_${rprefix}r${n_resamples}_${sn}
        python local/speaking_predictions_to_report_spk.py  --merged_speaker --data_dir $new_data_dir \
                                                    --result_root $new_runs_root/$exp_tag \
                                                    --all_bins "$all_bins" \
                                                    --cefr_bins "$cefr_bins" \
                                                    --folds "$folds" \
                                                    --question_type tb${test_book}p${part} \
                                                    --scores "$score_names" > $new_runs_root/$exp_tag/report_spk.log
    done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then  
    echo $runs_root/$exp_tag
    for sn in $score_names; do
        new_data_dir=${data_dir}_${rprefix}r${n_resamples}_${sn}
        new_runs_root=${runs_root}_${rprefix}r${n_resamples}_${sn}
        python local/visualization.py   --result_root $new_runs_root/$exp_tag \
                                    --all_bins "$all_bins" \
                                    --cefr_bins "$cefr_bins" \
                                    --affix "_spk" \
                                    --scores "$sn"
    done
fi
