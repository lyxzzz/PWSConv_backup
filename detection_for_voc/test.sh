#!/bin/bash
if [ $# -gt 0 ]
then
    prefix=$1
else
    prefix=last
fi

. ./utils.sh
runmodel(){
    if [ $dataset == "coco" ]
    then
        pyfile=test_coco.py
    else
        pyfile=test.py
    fi
    python ${pyfile} \
        --model=$test_model --ckpt_path=checkpoints --ckpt_name=${ckpt_name}_${prefix}.ckpt --postprocessing=ssdnms --save_file=result/result.txt --exp_name=${ziptime}_${expname} \
        --last_layer_norm=$last_norm --conv_type=$conv_type --norm_type=$norm_type --pwsepsilon=$pwsepsilon --gpuid=$gpuid
}

ckpt_name=ssd
dataset=voc
test_model=${dataset}_ssd
gpuid=0
memory_fraction=0.99
ziptime=`date +%Y%m%d%H`

pwsepsilon=0.001
last_norm=None
expname=PWS_${prefix}
conv_type=PWS
norm_type=None
batch_size=16
learning_rate=10.0
runmodel
