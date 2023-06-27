#!/bin/bash
code=12
U=0.
ave_ng=0.6
omega=0.01
nt=501
El_start=0.2
Nel=1
Nphi=20
storage=~/JJArrays/data
#length=9
for ave_ng in 0.2 0.3 0.5 0.7 0.8
do 
code=$(($code+1))

python main.py --code $code --U $U --ave_ng $ave_ng --omega $omega --nt $nt --Nel $Nel --Nphi $Nphi --El_start $El_start --path_name $storage --qinf --qephi
done
