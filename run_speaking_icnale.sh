
hostname=`hostname`
stage=0
stop_stage=1000
# data-related
corpus_dir="../corpus/speaking/ICNALE"
score_names="holistic"
kfold=1
# model-related
model=pool
exp_tag=bert-pool
model_path=bert-base-uncased
max_score=5
max_seq_length=256
num_epochs=15
score_loss=mse
stt_model_name=whisperv2_large
trans_type=trans_stt
extra_options=

. ./path.sh
. ./parse_options.sh

set -euo pipefail

anno_fn="annotations_${stt_model_name}.xlsx"

folds=`seq 1 $kfold`
data_dir=data-speaking/icnale/${trans_type}_${stt_model_name}
exp_root=exp-speaking/icnale/${trans_type}_${stt_model_name}
runs_root=runs-speaking/icnale/${trans_type}_${stt_model_name}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then  
    if [ -d $data_dir ]; then
        echo "[NOTICE] $data_dir is already existed."
        echo "Skip data preparation."
        sleep 5
    else
        python local/convert_icnale_to_aetg_data.py \
                    --corpus_dir $corpus_dir \
                    --data_dir $data_dir \
                    --anno_fn $anno_fn \
                    --text_column "$trans_type" \
                    --scores "$score_names" \
                    --kfold $kfold \
                    $extra_options
    fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
     
    for sn in $score_names; do
        for fd in $folds; do
            # model_args_dir
            output_dir=$exp_tag/${sn}/${fd}
            python3 run_speech_grader.py --do_train --save_best_on_evaluate --save_best_on_train \
                                         --do_lower_case \
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
            
            python3 run_speech_grader.py --do_test --model $model \
                                         --do_lower_case \
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
    python local/speaking_predictions_to_report_icnale.py  --data_dir $data_dir \
                                                    --result_root $runs_root/$exp_tag \
                                                    --folds "$folds" \
                                                    --scores "$score_names" > $runs_root/$exp_tag/report.log
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then  
    echo $runs_root/$exp_tag
    python local/visualization_icnale.py   --result_root $runs_root/$exp_tag \
                                    --scores "$score_names"
fi
