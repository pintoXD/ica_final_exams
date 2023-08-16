#!/bin/bash
nohup octave ps_com_pca.m > ps.out &
nohup octave ps_comite_com_pca.m > ps_comite.out &
nohup octave mlp_com_pca.m > mlp.out &
nohup octave mlp_comite_com_pca.m > mlp_comite.out &

sleep 5

jobs

