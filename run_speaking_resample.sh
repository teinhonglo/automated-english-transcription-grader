hostname=`hostname`
stage=0
stop_stage=1000
# data-related
corpus_dir="../corpus/speaking/GEPT_B1"
#score_names="content pronunciation vocabulary"
score_names="content"
anno_fn="new_111年口說語料.xlsx"
kfold=5
folds=`seq 1 $kfold`
part=2
test_on_valid="true"
merge_below_b1="false"
trans_type="trans_stt"
do_round="true"
n_resamples=100
# model-related
model=pool          # auto
exp_tag=bert-pool  # bert-model transformers=4.3.3, tokenizers=0.10.3
model_path=bert-base-uncased # bert-base-uncased
max_score=8
max_seq_length=512
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
extra_options=

. ./path.sh
. ./parse_options.sh


data_dir=data-speaking/gept-p${part}/$trans_type
exp_root=exp-speaking/gept-p${part}/$trans_type
runs_root=runs-speaking/gept-p${part}/$trans_type

if [ "$test_on_valid" == "true" ]; then
    extra_options="--test_on_valid"
    data_dir=${data_dir}_tov
    exp_root=${exp_root}_tov
    runs_root=${runs_root}_tov
fi

if [ "$do_round" == "true" ]; then
    extra_options="$extra_options --do_round"
    data_dir=${data_dir}_round
    exp_root=${exp_root}_round
    runs_root=${runs_root}_round
fi

if [ "$merge_below_b1" == "true" ]; then
    extra_options="$extra_options --merge_below_b1"
    data_dir=${data_dir}_bb1
    exp_root=${exp_root}_bb1
    runs_root=${runs_root}_bb1
fi

exp_root=${exp_root}
runs_root=${runs_root}


set -euo pipefail

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then  
    if [ -d $data_dir ]; then
        echo "[NOTICE] $data_dir is already existed."
        echo "Skip data preparation."
        sleep 5
    else
        python local/convert_gept_b1_to_aetg_data.py \
                    --corpus_dir $corpus_dir \
                    --data_dir $data_dir \
                    --anno_fn $anno_fn \
                    --id_column "編號名" \
                    --text_column "$trans_type" \
                    --scores "$score_names" \
                    --sheet_name $part \
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
                                         --do_lower_case --overwrite_cache\
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
            model_dir=$model_args_dir/best_train
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
                                                        --folds "$folds" \
                                                        --scores "$score_names" > $new_runs_root/$exp_tag/report.log
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then  
    for sn in $score_names; do
        new_data_dir=${data_dir}_${rprefix}r${n_resamples}_${sn}
        new_runs_root=${runs_root}_${rprefix}r${n_resamples}_${sn}
        echo $new_runs_root/$exp_tag

        python local/visualization.py   --result_root $new_runs_root/$exp_tag \
                                        --scores "$sn"
    done
fi
