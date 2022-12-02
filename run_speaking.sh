
hostname=`hostname`
stage=0
stop_stage=1000
# data-related
corpus_dir="../corpus/speaking/GEPT_B1"
score_names="content pronunciation vocabulary"
anno_fn="new_111年口說語料.xlsx"
# trainin-related
kfold=5
folds=`seq 1 $kfold`
# model-related
model_type=bert-model
model_path=bert-base-uncased
max_score=8
max_seq_length=512
part=3
test_on_valid="true"
merge_below_b1="false"
trans_type="trans_stt"
do_round="false"
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

set -euo pipefail

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then  
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


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
     
    for sn in $score_names; do
        for fd in $folds; do
            # model_args_dir
            output_dir=$model_type/${sn}/${fd}
            python3 run_speech_grader.py --do_train --save_best_on_evaluate --save_best_on_train \
                                         --do_lower_case \
                                         --model bert \
                                         --model_path bert-base-uncased \
                                         --num_train_epochs 6 \
                                         --logging_steps 20 \
                                         --gradient_accumulation_steps 1 \
                                         --max_seq_length $max_seq_length \
                                         --max_score $max_score --evaluate_during_training \
                                         --output_dir $output_dir \
                                         --score_name $sn \
                                         --data_dir $data_dir/$fd \
                                         --runs_root $runs_root \
                                         --exp_root $exp_root
        done
    done
fi



if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then  
    
    for sn in $score_names; do
        for fd in $folds; do
            output_dir=$model_type/${sn}/${fd}
            model_args_dir=$model_type/${sn}/${fd}
            model_dir=$model_args_dir/best_train
            predictions_file="$runs_root/$output_dir/predictions.txt"
            
            python3 run_speech_grader.py --do_test --model bert \
                                         --do_lower_case \
                                         --model_path bert-base-uncased \
                                         --model_args_dir $model_args_dir \
                                         --max_seq_length $max_seq_length \
                                         --max_score $max_score \
                                         --model_dir $model_dir \
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
    python local/speaking_predictions_to_report.py  --data_dir $data_dir \
                                                    --result_root $runs_root/$model_type \
                                                    --folds "$folds" \
                                                    --scores "$score_names" > $runs_root/$model_type/report.log
fi
