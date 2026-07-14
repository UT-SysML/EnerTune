#!/bin/bash

source ../helper.sh

PYTHON=${VENV}/bin/python3
NCU=/usr/local/cuda/bin/ncu
NSYS=/usr/local/bin/nsys

run_decorated_inference() {
    pre=$1
    post="> /dev/null 2>&1"
    if [[ $# -eq 2 ]]; then
        post=$2
    fi

    command="${pre} \
        ${PYTHON} ../src/executor.py \
        --device-id ${device_id} \
        --model ${model} \
        --batch-size ${batch} \
        --distribution-type closed \
        --rps 0 \
        --tid 0 \
        --num-reqs 1 \
        ${post} &"
    # echo $command
    eval "$command"
    profile_pid=$!
    sleep 10
    readarray -t forked_pids < <(ps -eaf | grep executor.py | grep -v "${NCU}" | grep -v "nsight" | grep -v grep | grep "${model}" | awk '{print $2}')
    echo "Forked PIDs: ${forked_pids[@]}"
    if [[ ${#forked_pids[@]} != 1 ]]; then
        echo "Expected 1 executor.py process! Seen != 1..."
        echo "Inspect using command: ' ps -eaf | grep executor.py.py | grep -v "${NSYS}" | grep -v grep'"
        return 1
    fi

    # Wait for the model to load
    for pid in "${forked_pids[@]}"
    do
        while :
        do
            if [[ -f /tmp/${pid} ]]; then
                lt="$lt, $(cat /tmp/${pid})"
                ${SUDO} rm -f /tmp/${pid}
                loaded_procs+=(${pid})
                break
            elif [[ -f /tmp/${pid}_oom ]]; then
                ${SUDO} rm -f /tmp/${pid}_oom
                break
            fi

        done
    done

    # Start inference
    ${SUDO} kill -SIGUSR1 ${forked_pids[@]}


    # Wait till the prefixed command completes
    while ${SUDO} kill -0 ${profile_pid} >/dev/null 2>&1; do sleep 1; done
        
}

profile_model() {
    device_type=$1
    device_id=$2
    model=$3
    batch=$4
    model_type=$5

    if [[ (${device_type} != "v100" && ${device_type} != "a100" && ${device_type} != "h100") ]]; then
        echo "Invalid device_type: ${device_type}"
        print_help
        exit 1
    fi

    if [[ -z ${device_id} || ! ${device_id} =~ ^[0-9]+$ ]]; then
        echo "Invalid device_id: ${device_id}"
        print_help
        exit 1
    fi

    # Add logic for checking model is valid model we support

    result_dir=$(pwd)/data/${device_type}/${model}
    mkdir -p ${result_dir}

    echo 'NCU Profiling...'
    run_decorated_inference "${SUDO} -E ${NCU} -f --csv --metrics "l1tex__m_xbar2l1tex_read_bytes.sum.per_second,l1tex__m_l1tex2xbar_write_bytes.sum.per_second,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second" --nvtx --nvtx-include "start/" -o ${result_dir}/batchsize_${batch}_output_ncu.csv"

    # echo 'NSYS Profiling...'
    # run_decorated_inference "${SUDO} -E env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" ${NSYS} profile --trace=cuda,nvtx,osrt,cudnn,cublas --sample=none --gpu-metrics-device=0 --gpu-metrics-frequency=50000 --stats=true --force-overwrite=true --stop-on-exit=true --wait=all -o ${result_dir}/batchsize_${batch}_output_nsys --show-output=true"
}



if [[ $# -lt 4 ]]; then
    echo "Expected Syntax: '$0 <v100 | a100 | h100> <device_id> model batchsize <vision | bert | transformer>"
    echo "Examples:"
    echo " $0 v100 0 vgg11 16 vision"
    exit 1
fi

profile_model $@
