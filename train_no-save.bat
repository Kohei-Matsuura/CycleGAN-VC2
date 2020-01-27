#!/bin/bash

S=.

mkdir /n/rd33/matsuura/save/${S}
CUDA_VISIBLE_DEVICES=3 python train.py '/n/rd33/matsuura/evals/KS.over128.htks' '/n/rd33/matsuura/evals/KM.over128.htks' /n/rd33/matsuura/save/${S}
