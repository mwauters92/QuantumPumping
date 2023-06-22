#!/bin/bash
code=9
U=0.0
ave_ng=0.4
omega=0.01
nt=301 #501
Nel=10 #15
Nphi=15 #20
storage=~/JJArrays/data
length=9

python main.py --code $code --U $U --ave_ng $ave_ng --omega $omega --nt $nt --Nel $Nel --Nphi $Nphi --path_name $storage --n_sites $length --qinf --qephi
