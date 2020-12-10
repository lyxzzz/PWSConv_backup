#!/bin/bash
remakedir(){
    if [ -d $1 ]
    then
        rm -rf $1
        mkdir $1
    else
        mkdir $1
    fi
}

createdir(){
    if [ ! -d $1 ]
    then
        mkdir $1
    fi
}

configconv(){
    createdir logs
    createdir checkpoints
    createdir historycode
    createdir historyresult
    zipname=historycode/$1.zip
    zip -r $zipname *.py python result test_cfgs train_cfgs .run .restore *.sh
    if [ "$2" = "y" ]
    then
        echo "RM FILES"
        remakedir logs
        remakedir checkpoints
    else
        echo "NOT RM FILES"
    fi
}