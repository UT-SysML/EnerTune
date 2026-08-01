#!/bin/bash

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}/systems/ener-tune/estimator-ablation/load-25
./gpu0.sh
./gpu1.sh
./gpu2.sh
./gpu3.sh
mkdir -p ${git_dir}/results/et-pwr-estimator-ablation-results/load-25
cp -r ${git_dir}/results/a100/* ${git_dir}/results/et-pwr-estimator-ablation-results/load-25/
rm -rf ${git_dir}/results/a100/*
