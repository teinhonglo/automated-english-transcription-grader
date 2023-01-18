
hostname=`hostname`
stage=0
stop_stage=1000
# data-related
corpus_dir="/share/nas167/teinhonglo/AcousticModel/spoken_test/corpus/writing/GSAT"
#score_names="content organization grammar vocabulary"
score_names="content"
anno_fn="109年寫作語料.xlsx"
kfold=5
folds=`seq 1 $kfold`
test_on_valid="true"
trans_type="origin"
do_l2r="false"
# model-related
model=pool
model_type=electra-base-discriminator
model_path=google/electra-base-discriminator
score_loss="mse"
num_epochs=6
max_score=8
max_seq_length=512
extra_options=""
# training-related
warmup_steps=150  # 0
max_grad_norm=10   # 1.0
learning_rate=5e-5  # 5e-5
eval_mode="final"

exp_tag="reg_mseloss_meanpool_warmup150_reinit2_clip10_mlm0.1word0.2"

. ./path.sh
. ./parse_options.sh

data_dir=data-writting/gsat109/$trans_type
exp_root=exp-writting/gsat109/$trans_type
runs_root=runs-writting/gsat109/$trans_type

if [ "$test_on_valid" == "true" ]; then
    extra_options="--test_on_valid"
    data_dir=${data_dir}_tov
    exp_root=${exp_root}_tov
    runs_root=${runs_root}_tov
fi

if [ "$do_l2r" == "true" ]; then
    exp_root=${exp_root}_l2r
    runs_root=${runs_root}_l2r
fi

exp_root=${exp_root}_${exp_tag}
runs_root=${runs_root}_${exp_tag}

set -euo pipefail

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then  
    if [ -d $data_dir ]; then
        echo "[NOTICE] $data_dir is already existed."
        echo "Skip data preparation."
        sleep 5
    else
        
        python local/convert_gsat_to_aetg_data.py \
                        --corpus_dir $corpus_dir \
                        --data_dir $data_dir \
                        --anno_fn $anno_fn \
                        --id_column "編號名" \
                        --text_column "作文(工讀生校正)" \
                        --scores "$score_names" \
                        --kfold $kfold $extra_options
    fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
     
    for sn in $score_names; do
        for fd in $folds; do
            output_dir=$model_type/${sn}/${fd}
            python3 run_speech_grader.py --do_train --save_best_on_evaluate --save_best_on_train \
                                         --do_lower_case \
                                         --model $model \
                                         --model_path $model_path \
                                         --num_train_epochs $num_epochs \
                                         --gradient_accumulation_steps 1 \
                                         --max_grad_norm $max_grad_norm \
                                         --max_seq_length $max_seq_length \
                                         --warmup_steps $warmup_steps \
                                         --learning_rate $learning_rate \
                                         --max_score $max_score --evaluate_during_training \
                                         --use_mlm_objective \
                                         --mlm_alpha 0.1 \
                                         --score_alpha 0.9 \
                                         --score_loss $score_loss \
                                         --output_dir $output_dir \
                                         --score_name $sn \
                                         --data_dir $data_dir/$fd \
                                         --runs_root $runs_root \
                                         --exp_root $exp_root
                                         #--warmup_step 1320 \
        done
    done
fi



if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then  
    
    for sn in $score_names; do
        for fd in $folds; do
            output_dir=$model_type/${sn}/${fd}
            model_args_dir=$model_type/${sn}/${fd}
            model_dir=$model_args_dir/${eval_mode}
            predictions_file="$runs_root/$output_dir/predictions.txt"
            
            python3 run_speech_grader.py --do_test --model $model \
                                         --do_lower_case \
                                         --model_path $model_path \
                                         --model_args_dir $model_args_dir \
                                         --max_seq_length $max_seq_length \
                                         --max_score $max_score \
                                         --model_dir $model_dir \
                                         --use_mlm_objective \
                                         --mlm_alpha 0.1 \
                                         --score_alpha 0.9 \
                                         --predictions_file $predictions_file \
                                         --data_dir $data_dir/$fd \
                                         --score_name $sn \
                                         --runs_root $runs_root \
                                         --output_dir $output_dir \
                                         --exp_root $exp_root
        done
    done 
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then  
    python local/writing_predictions_to_report.py   --data_dir $data_dir \
                                                    --result_root $runs_root/$model_type \
                                                    --folds "$folds" \
                                                    --scores "$score_names" > $runs_root/$model_type/report.log
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then  
    python local/visualization.py   --result_root $runs_root/$model_type \
                                    --scores "$score_names"
fi
