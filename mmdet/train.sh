#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#172.29.7.223
EXPNAME=PWS_Scratch_2x
MODEL_TYPE=maskrcnn
CONFIG_FILE=configs/${MODEL_TYPE}.py
WORK_DIR=result/${EXPNAME}

rm -rf ${WORK_DIR}

#b mmdet/models/backbones/resnet
python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --gpus 1

DATETIME=`date "+%Y-%m-%d-%H"`
zip -r result/backup/${DATETIME}_${EXPNAME}.zip ${WORK_DIR}/epoch_final.pth ${WORK_DIR}/log.txt ${WORK_DIR}/events* configs/${MODEL_TYPE}.py
