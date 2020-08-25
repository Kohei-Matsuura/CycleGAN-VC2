#!/bin/bash

mkdir /n/rd33/matsuura/save/${1}
mkdir /n/rd33/matsuura/save/${1}/params
gpujob python train.py '/n/rd33/matsuura/evals/cgan/KS.over128.htks' '/n/rd33/matsuura/evals/KM.over128.htks' /n/rd33/matsuura/save/${1} | tee /n/rd33/matsuura/save/${1}/loss.log
