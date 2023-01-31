
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
# model-related
model=pool
exp_tag=bert-pool-model
model_path=bert-base-uncased
max_score=8
max_seq_length=128
num_epochs=6
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


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then  
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

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
     
    for sn in $score_names; do
        for fd in $folds; do
            # model_args_dir
            output_dir=$exp_tag/${sn}/${fd}
            python3 run_speech_grader.py --do_train --save_best_on_evaluate --save_best_on_train \
                                         --do_lower_case --overwrite_cache \
                                         --model $model \
                                         --model_path $model_path \
                                         --num_train_epochs $num_epochs \
                                         --logging_steps 20 \
                                         --gradient_accumulation_steps 1 \
                                         --max_seq_length $max_seq_length \
                                         --max_score $max_score --evaluate_during_training \
                                         --output_dir $output_dir \
                                         --score_name $sn \
                                         --score_loss $score_loss \
                                         --data_dir $data_dir/$fd \
                                         --runs_root $runs_root \
                                         --exp_root $exp_root
        done
    done
fi



if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then  
    
    for sn in $score_names; do
        for fd in $folds; do
            output_dir=$exp_tag/${sn}/${fd}
            model_args_dir=$exp_tag/${sn}/${fd}
            model_dir=$model_args_dir/best
            predictions_file="$runs_root/$output_dir/predictions.txt"
            
            python3 run_speech_grader.py --do_test --overwrite_cache --model $model \
                                         --do_lower_case --overwrite_cache \
                                         --model_path $model_path \
                                         --model_args_dir $model_args_dir \
                                         --max_seq_length $max_seq_length \
                                         --max_score $max_score \
                                         --model_dir $model_dir \
                                         --predictions_file $predictions_file \
                                         --data_dir $data_dir/$fd \
                                         --score_name $sn \
                                         --score_loss $score_loss \
                                         --runs_root $runs_root \
                                         --output_dir $output_dir \
                                         --exp_root $exp_root
        done
    done 
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then  
    python local/speaking_predictions_to_report.py  --data_dir $data_dir \
                                                    --result_root $runs_root/$exp_tag \
                                                    --all_bins "$all_bins" \
                                                    --cefr_bins "$cefr_bins" \
                                                    --folds "$folds" \
                                                    --scores "$score_names" > $runs_root/$exp_tag/report.log
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then  
    echo $runs_root/$exp_tag
    python local/visualization.py   --result_root $runs_root/$exp_tag \
                                    --all_bins "$all_bins" \
                                    --cefr_bins "$cefr_bins" \
                                    --scores "$score_names"
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then  
    python local/speaking_predictions_to_report_spk.py  --merged_speaker --data_dir $data_dir \
                                                    --result_root $runs_root/$exp_tag \
                                                    --all_bins "$all_bins" \
                                                    --cefr_bins "$cefr_bins" \
                                                    --folds "$folds" \
                                                    --question_type tb${test_book}p${part} \
                                                    --scores "$score_names" > $runs_root/$exp_tag/report_spk.log
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then  
    echo $runs_root/$exp_tag
    python local/visualization.py   --result_root $runs_root/$exp_tag \
                                    --all_bins "$all_bins" \
                                    --cefr_bins "$cefr_bins" \
                                    --affix "_spk" \
                                    --scores "$score_names"
fi