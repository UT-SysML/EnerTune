#!/bin/bash -e

print_help() {
    echo "Usage: ${0} [OPTIONS] model1-parameter model2-parameter ..."
    echo "Options:"
    echo "  --device-type   DEVICE_TYPE                   v100, a100, h100                                   (required)"
    echo "  --run-id        UNIQUE RUN ID                 1                                                  (required)"
    echo "  --load          LOAD_INDICATOR                0.5, 0.2, 1                                        (default 1)"
    echo "  --min-freq      MIN_GPU_FREQ                  1410, increments of 15 MHz                         (default 1410 MHz)"
    echo "  --max-freq      MAX_GPU_FREQ                  1410, increments of 15 MHz                         (default 1410 MHz)"

    echo "  -h, --help                                    Show this help message"
    echo -e "\n"

    echo "NOTE:"
    echo "  Load generation is done via Distribution Type and RPS"
    echo -e "\n"

    echo "Example:"
    echo " $0 --device-type a100 --load 1 --run-id 1 '[{\"model\": \"vgg19\", \"batch-size\": 32, \"distribution-type\": \"poisson\", \"rps\": 40}, {\"model\": \"mobilenet_v2\", \"batch-size\": 4, \"distribution-type\": \"closed\", \"rps\": 0}]'"
    echo ""
    echo " $0 --device-type a100 --load 1 --run-id 1 '[{\"model\": \"vgg19\", \"batch-size\": 32, \"distribution-type\": \"poisson\", \"rps\": 40, \"mig-slice\": 4}, {\"model\": \"mobilenet_v2\", \"batch-size\": 4, \"distribution-type\": \"closed\", \"rps\": 0}, \"mig-slice\": 3]'"
    echo -e "\n"

    echo "NOTE: MIG must be enabled | disabled explicitly followed by a reboot on an A100"
    echo "--> nvidia-smi -i 0 -mig ENABLED (OR) nvidia-smi -i 0 -mig DISABLED"
    echo "--> reboot"
}

get_input() {
    # Parse arguments
    load=1000
    min_freq=1410
    max_freq=1410
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device-type)
                device_type="$2"
                shift 2
                ;;
            --load)
                load="$2"
                shift 2
                ;;
            --run-id)
                run_id="$2"
                shift 2
                ;;
            --min-freq)
                min_freq="$2"
                shift 2
                ;;
            --max-freq)
                max_freq="$2"
                shift 2
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            *)
                parse_model_parameters "$@"
                shift $#
            ;;
        esac
    done
    num_procs=${#model_run_params[@]}
}

validate_input() {
    if [[ (${device_type} != "v100" && ${device_type} != "a100" && ${device_type} != "h100") ]]; then
        echo "Invalid device_type: ${device_type}"
        print_help
        exit 1
    fi

    if [[ -z ${load} || ! ${load} =~ ^[+-]?[0-9]*\.?[0-9]+$ ]]; then
        echo "load must be a float. Got ${load}"
        print_help
        exit 1
    fi

    if [[ -z ${run_id} ]]; then
        echo "run-id is a required argument"
        print_help
        exit 1
    fi

    if [[ ${num_procs} -eq 0 ]]; then
        echo "Need at least 1 model configuration to run"
        print_help
        exit 1
    fi

    if [[ ! -z ${min_freq} ]]; then
        remainder=$(( (1410 - min_freq) % 15 ))
        if [[ ${remainder} -ne 0 ]]; then
            echo "min_freq must satisfy: (1410-min_freq) mod 15 = 0"
            echo "Got min_freq=${min_freq}, remainder=${remainder}"
            print_help
            exit 1
        fi
    fi

    if [[ ! -z ${max_freq} ]]; then
        remainder=$(( (1410 - max_freq) % 15 ))
        if [[ ${remainder} -ne 0 ]]; then
            echo "max_freq must satisfy: (1410-max_freq) mod 15 = 0"
            echo "Got max_freq=${min_freq}, remainder=${remainder}"
            print_help
            exit 1
        fi
    fi
}

modes_ran=()
device_ids_ran=()
uuids_ran=()
cleanup_handler() {
    original_x=$(set +o | grep xtrace)
    set -x
    local job_mix_exit_code=$?

    # Kill any pending procs
    if [[ ${#uuids_ran[@]} -gt 0 ]]; then
        IFS="|"
        uuid_grep="${uuids_ran[*]}"
        unset IFS
        ps -eaf | grep executor.py | egrep "${uuid_grep}" |
            grep -v grep | awk '{print $2}' |
            xargs -I{} kill -9 {} || :
    fi

    # Clean the modes ran
    for ((y=0; y<${#modes_ran[@]}; y++))
    do
        echo "Cleaning up for ${modes_ran[$y]} on ${device_ids_ran[$y]}"
        cleanup ${modes_ran[$y]} ${device_ids_ran[$y]} || :
    done

    # Clean up the fifo pipe created
    rm -f ${fifo_pipe} || :

    # Clean up the IPC queue
    ipcrm --all=msg || :

    # Exit with the exit code with which the handler was called
    echo "Exiting with Error code: ${job_mix_exit_code} (0 is clean exit)"
    eval "$original_x"
    exit ${job_mix_exit_code}

}

setup_expr() {
    trap cleanup_handler EXIT

    # Find where to store reults
    get_result_dir models[@] batch_sizes[@] distribution_types[@] ${device_type}
    mkdir -p ${result_dir}

    # Create a FIFO to listen on
    fifo_pipe=/tmp/${run_id}
    rm -f ${fifo_pipe}
    mkfifo ${fifo_pipe}

    # Clean up the IPC queue
    ipcrm --all=msg || :
}

read_fifo() {
    pipe_name=$1
    read json_data < ${pipe_name} # blocking
    echo "Got: ${json_data}"
    mode_to_run=$(echo "$json_data" | jq -r '.mode')
    device_id_to_run=$(echo "$json_data" | jq -r '.["device-id"]')
    modes_ran+=(${mode_to_run})
    device_ids_ran+=(${device_id_to_run})
}

compute_stats() {
    declare -a pkl_files_arg=("${!1}")
    local mode_arg=$2
    local result_dir_arg=$3
    local load=$4
    local min_freq=$5
    local max_freq=$6

    cmd="python3 src/stats.py \
        --mode ${mode_arg} \
        --load ${load} \
        --min-freq ${min_freq} \
        --max-freq ${max_freq} \
        --result-dir ${result_dir_arg} \
        ${pkl_files_arg[@]}"
    eval $cmd
    echo "Results stored in: ${result_dir_arg}"
}

nvml_pid=-1
start_expr() {
    local mode_arg=$1
    local device_id_arg=$2
    local uuid_arg=$3
    local run_id_arg=$4
    local min_freq=$5
    local max_freq=$6

    assert_mig_status ${mode_arg} ${device_id_arg}
    enable_mps_if_needed ${mode_arg} ${device_id_arg}

    setup_mig_if_needed ${mode_arg} ${device_id_arg} ${num_procs} ${mig_slices[0]} mig_slices


    # if [[ ${mode_arg} == "custom-mig" ]]; then
    #     for (( c=0; c<${num_procs}; c++ ))
    #     do
    #         setup_mig_if_needed mig ${device_id_arg} ${num_procs} ${mig_slices[$c]}
    #     done
    # else
    #     setup_mig_if_needed ${mode_arg} ${device_id_arg} ${num_procs} ${mig_slices[0]}
    # fi

    # setup_mig_if_needed ${mode_arg} ${device_id_arg} ${num_procs} ${mig_slices[0]}
    if [[ ${min_freq} -gt 0 && ${max_freq} -gt 0 ]]; then
        set_gpu_freq ${device_id_arg} ${min_freq} ${max_freq}
    fi
    # set_gpu_freq ${device_id_arg} ${min_freq} ${max_freq}

    # Get MPS capping percentage values if we are using MPS capping mode
    if [[ ${mode_arg} == "mps-miglike" ]]; then
        percent=($(echo ${mps_mig_percentages[$num_procs]} | tr "," "\n"))
    elif [[ ${mode_arg} == "mps-equi" ]]; then
        percent=($(echo ${mps_equi_percentages[$num_procs]} | tr "," "\n"))
    elif [[ ${mode_arg} == "mps-manual-cap" || ${mode_arg} == "mps-throttle" ]]; then
        percent=("${mig_slices[@]}")
    fi

    mig_info=$(awk "/^GPU ${device_id_arg}:/{p=1; next} /^GPU/{p=0} p && /^  MIG/{print}" <<< "$(nvidia-smi -L)")
    chunk_id=($(echo "$mig_info" | awk '{print $2}'))
    cci_uuid=($(echo "$mig_info" | awk '{print $NF}' | sed 's/[()]//g'))
    cmd_arr=()
    
    # Launch NVML C++ daemon here
    nvml_cmd="./profiler/gmonitor ${device_id_arg} ${result_dir}/energy-${mode_arg}${mig_slices[0]}-${max_freq}.csv &"
    echo "${nvml_cmd}"
    eval $nvml_cmd
    nvml_pid=$!

    for (( c=0; c<${num_procs}; c++ ))
    do
        if [[ ${mode_arg} == "mig" || ${mode_arg} == "mig-profile-1" || ${mode_arg} == "mig-profile-2" || ${mode_arg} == "mig-profile-3" || ${mode_arg} == "custom-mig" ]]; then
            echo "Setting ${chunk_id[$c]} for ${models[$c]} with cci_uuid ${cci_uuid[$c]}"
            export_prefix="export CUDA_VISIBLE_DEVICES=${cci_uuid[$c]}"
        elif [[ ${mode_arg} == "mps-miglike" || ${mode_arg} == "mps-equi" || ${mode_arg} == "mps-manual-cap" ]]; then
            echo "Setting ${percent[$c]}% for ${models[$c]}"
            export_prefix="export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1 && \
                           export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${percent[$c]} && \
                           export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${device_id_arg} && \
                           export CUDA_VISIBLE_DEVICES=0"
        elif [[ ${mode_arg} == "mps-uncap" ]]; then
            export_prefix="export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=0 && \
                           export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${device_id_arg} && \
                           export CUDA_VISIBLE_DEVICES=0"
        elif [[ ${mode_arg} == "mps-throttle" ]]; then
            export_prefix="export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=0 && \
                           export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${device_id_arg} && \
                           export CUDA_VISIBLE_DEVICES=0 && \
                           export GPU_THROTTLE_GROUP=gpu${device_id_arg} && \
                           export GPU_THROTTLE_SHARE=${percent[$c]} && 
                           export GPU_THROTTLE_WINDOW_MS=100 && 
                           export LD_PRELOAD=${HOME}/sus-gpus/throttler/libgpucthrottle.so"
        else
            export_prefix="export CUDA_VISIBLE_DEVICES=${device_id_arg}"
        fi  


        # throttle_env="export GPU_THROTTLE_GROUP=gpu${device_id_arg} && \
        #               export GPU_THROTTLE_SHARE=0.6 && \
        #               export GPU_THROTTLE_WINDOW_MS=100 && \
        #               export LD_PRELOAD=${HOME}/sus-gpus/throttlerlibgpucthrottle.so"
        # throttle_env="export STUPID=0"
        

        # Assuming we can run 7 models in parallel in a device
        cpu=$(((device_id_arg * 7) + (c+1)))

        # We always set device-id to 0, as CUDA_VISIBLE_DEVICES exports only 1 GPU per model
        cmd="${export_prefix} && \
            taskset -c ${cpu} python3 src/executor.py \
            --device-id 0 \
            ${model_run_params[$c]} \
            --run-id ${run_id_arg} \
            --tid ${c} \
            --uuid ${uuid_arg} > /dev/null &"
	echo ${cmd}
        if [[ "${models[$c]}" == "gemm" ]]; then
            batch_size=${batch_sizes[$c]}
            cmd="${export_prefix} && \
                taskset -c ${cpu} ./src/gemm ${batch_size} ${batch_size} ${batch_size} > /dev/null &"
        elif [[ "${models[$c]}" == "gemv" ]]; then
            batch_size=${batch_sizes[$c]}
            cmd="${export_prefix} && \
                taskset -c ${cpu} ./src/gemv ${batch_size} ${batch_size} > /dev/null &"
        elif [[ "${models[$c]}" == "saxpy" ]]; then
            batch_size=${batch_sizes[$c]}
            cmd="${export_prefix} && \
                taskset -c ${cpu} ./src/saxpy ${batch_size} 350000 65536 > /dev/null &"
        fi
        eval $cmd
        cmd_arr+=("${cmd}")

    done

    # Check if processes are alive
    readarray -t executor_forked_pids < <(ps -eaf | grep executor.py |
        grep ${uuid_arg} | grep -v grep |
        awk '{for (i=1; i<=NF; i++) if ($i == "--tid") print $(i+1),$0}' |
        sort -n | cut -d' ' -f2- | awk '{print $2}')
    readarray -t gemm_forked_pids < <(ps -eaf | grep ./src/gemm | grep -v grep | sort -n | cut -d' ' -f2- | awk '{print $2}')
    readarray -t gemv_forked_pids < <(ps -eaf | grep ./src/gemv | grep -v grep | sort -n | cut -d' ' -f2- | awk '{print $2}')
    readarray -t saxpy_forked_pids < <(ps -eaf | grep ./src/saxpy | grep -v grep | sort -n | cut -d' ' -f2- | awk '{print $2}')

    if [[ ${#executor_forked_pids[@]} -ne 0 ]]; then
        forked_pids=("${executor_forked_pids[@]}")
    elif [[ ${#gemm_forked_pids[@]} -ne 0 ]]; then
        forked_pids=("${gemm_forked_pids[@]}")
    elif [[ ${#gemv_forked_pids[@]} -ne 0 ]]; then
        forked_pids=("${gemv_forked_pids[@]}")
    elif [[ ${#saxpy_forked_pids[@]} -ne 0 ]]; then
        forked_pids=("${saxpy_forked_pids[@]}")
    else
        forked_pids=()
    fi
    
    if [[ ${#forked_pids[@]} -ne ${num_procs} ]]; then
        echo "Expected ${num_procs} processes. But found ${#forked_pids[@]}}"
        echo "Examine commands: "
        for cmd in "${cmd_arr[@]}"
        do
            echo "  ${cmd}"
        done
        exit 1
    fi

    # Wait til all pids have loaded their models
    
    lt=""
    loaded_procs=()
    echo ${forked_pids[@]}
    for pid in "${forked_pids[@]}"
    do
        load_ctr=10000
        while [[ ${load_ctr} -gt 0 ]];
        do
            if [[ ${#executor_forked_pids[@]} -gt 0 ]]; then
                if [[ -f /tmp/${pid} ]]; then
                    lt="$lt, $(cat /tmp/${pid})"
                    rm -f /tmp/${pid}
                    loaded_procs+=(${pid})
                    break
                fi
            else
                loaded_procs+=(${pid})
                break
            fi

            if ! kill -0 "${pid}" &> /dev/null; then
                echo "Process no longer alive"
                load_ctr=0
                break
            fi
            ((load_ctr--))
            sleep 0.25
        done

        if [[ ${load_ctr} -eq 0 ]]; then
            echo "Some of the models did not load!"
            echo "Examine commands: "
            for cmd in "${cmd_arr[@]}"
            do
                echo "  ${cmd}"
            done
            exit 1
        fi
    done

    # Touch to indicate models are located
    touch /tmp/${run_id_arg}.load

    # Start inference
    echo "Starting inference on ${loaded_procs[@]}"
    if [[ ${#executor_forked_pids[@]} -gt 0 ]]; then
        kill -SIGUSR1 ${loaded_procs[@]}
    fi
    # kill -SIGUSR1 ${loaded_procs[@]}
}

run_expr() {
    local procs=()
    local prev=()

    # Wait to be told which GPU and mechanism to run on
    read_fifo ${fifo_pipe}

    # Launch processes
    while [[ ${mode_to_run} != "stop" ]];
    do
        # Enable PM in GPU
        ${SUDO} nvidia-smi -i ${device_id_to_run} -pm ENABLED

        # Acquire lock on the GPU
        lock_gpu ${device_id_to_run}

        # Get a unique id for the run
        local uuid=$(uuidgen)
        uuids_ran+=(${uuid})

        # Begin the experiment
        start_expr ${mode_to_run} ${device_id_to_run} ${uuid} ${run_id} ${min_freq} ${max_freq}

        # Keep track of state
        prev=("${loaded_procs[@]}")
        procs+=( "${loaded_procs[@]}" )
        prev_mode_run=${mode_to_run}
        prev_device_id_run=${device_id_to_run}

        # If num_reqs is still default, block until the next command from control plane comes in
        if [[ ${num_reqs[0]} -eq 9223372 ]]; then
            read_fifo ${fifo_pipe}
        else
            # Poll until all processes complete if num_reqs is less than default
            if [[ ${num_reqs[0]} -lt 9223372 ]]; then
                running_procs=("${loaded_procs[@]}")
                while true; do
                    # Filter out non-existing PIDs
                    active_procs=()
                    for pid in "${running_procs[@]}"; do
                        if taskset -c 0 kill -0 ${pid} 2>/dev/null; then
                            active_procs+=("$pid")  # Keep running PIDs
                        fi
                    done

                    # Break the loop if no PIDs are active
                    if [[ ${#active_procs[@]} -eq 0 ]]; then
                        echo "All processes have exited."
                        break
                    fi
                    
                    running_procs=("${active_procs[@]}")
                    sleep 1
                done
                break
            fi
        fi
    done

    # Stop NVML process here
    sleep 3
    kill ${nvml_pid}

    # Make sure to stop all inferences
    safe_clean_gpu prev[@] ${prev_mode_run} ${prev_device_id_run}

    # Wait for the process to exit and get stats pkl file
    pkl_files=()
    for pid in "${procs[@]}"
    do
        while taskset -c 0 kill -0 ${pid} >/dev/null 2>&1; do sleep 1; done
        pkl_file="/tmp/${pid}.pkl"
        pkl_files+=(${pkl_file})
    done

    mode_run=${prev_mode_run}
    
    if [[ ${#pkl_files[@]} -gt 0 ]]; then
        compute_stats pkl_files[@] ${mode_run} ${result_dir} ${load} ${min_freq} ${max_freq}
    else
        echo "No modes were run, so not computing stats"
    fi

}

git_dir=$(git rev-parse --show-toplevel)
cd ${git_dir}
source helper.sh && helper_setup
get_input $@
validate_input
setup_expr
run_expr
