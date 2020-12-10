#!/bin/bash
. ./utils.sh
runmodel(){
    python test.py \
        --model=$test_model --ckpt_path=checkpoints --ckpt_name=${ckpt_name}_${prefix}.ckpt --postprocessing=ssdnms --save_file=result/result.txt --exp_name=${ziptime}_${expname} \
        --conv_type=$conv_type --norm_type=$norm_type --pwsepsilon=$pwsepsilon --gpuid=$gpuid
}

if [ $# -gt 0 ]
then
    prefix=$1
else
    prefix=last
fi
ckpt_name=vgg
gpuid=0
memory_fraction=0.99
ziptime=`date +%Y%m%d%H`

# ###########################start###########################
pwsepsilon=0.001
train_model=cifar_vgg
test_model=cifar_vgg
expname=PWS_${prefix}
conv_type=PWS
norm_type=None
batch_size=128
learning_rate=10.0
runmodel

