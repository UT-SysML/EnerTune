#!/bin/bash -e

log() {
    echo -e "$@"
    if [[ ! -z ${PRINT_OUTS} ]]; then
        echo -e "$@" >> ${PRINT_OUTS}
    fi
}

print_log_location() {
    local run_sh_exit_code=$?
    log "Exiting with error_code=${run_sh_exit_code} (0 is clean exit)"
    log "Examine ${PRINT_OUTS} for logs"
    exit ${run_sh_exit_code}
}

print_help() {
    log "Usage: ${0} [OPTIONS] model1-parameter model2-paramete ..."
    log "Options:"
    log "  --device-type   DEVICE_TYPE                   v100, a100, h100, 4090                             (required)"
    log "  --device-id     DEVICE_ID                     0, 1, 2, ..                                        (required)"
    log "  --modes         MODE1,MODE2,MODE3             mps-uncap, tm, mig                                 (default mps-uncap,tm,mig)"
    log "  --duration      DURATION_OF_EXPR_IN_SECONDS                                                      (default 0)"
    log "  --distribution  DISTRIBUTION_TYPE             poisson, closed, point                             (default closed)"
    # log "  --load-start    LOAD_START                    0.1                                                (default 1.0)"
    # log "  --load-end      LOAD_END                      1.5                                                (default 1.0)"
    # log "  --load-step     LOAD_STEP                     0.1                                                (default 0.1)"
    log "  --min-freq      MIN_GPU_FREQ                  1410, increments of 15 MHz                         (default 1410 MHz)"
    log "  --max-freq      MAX_GPU_FREQ                  1410, increments of 15 MHz                         (default 1410 MHz)"
    log "  -h, --help                                    Show this help message"
    log -e "\n"

    log "Examples:"
    log " $0 --device-type 4090 --device-id 0 -modes tm --duration 10 diffusion-1 <format: Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps>"
    log " $0 --device-type a100 --device-id 1 --modes tm --duration 20 diffusion-1 whisper-1 <format: Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps>"
    log -e "\n"
}

get_input() {
    model_run_params=()
    duration=0
    distribution=closed
    # load_start=1.0
    # load_end=1.0
    # load_step=0.1
    min_freq=1410
    max_freq=1410
    modes=("mps-uncap" "tm" "mig")
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device-type)
                device_type="$2"
                shift 2
                ;;
            --device-id)
                device_id="$2"
                shift 2
                ;;
            --duration)
                duration=$2
                shift 2
                ;;
            --distribution)
                distribution="$2"
                shift 2
                ;;
            # --load-start)
            #     load_start=$2
            #     shift 2
            #     ;;
            # --load-end)
            #     load_end=$2
            #     shift 2
            #     ;;
            # --load-step)
            #     load_step=$2
            #     shift 2
            #     ;;
            --modes)
                unset modes
                IFS=',' read -r -a modes <<< "$2"
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
                model_run_params+=("$1")
                shift
                ;;
        esac
    done
}

validate_input() {
    num_procs=${#model_run_params[@]}
    if [[ -z ${device_type} || (${device_type} != "v100" && ${device_type} != "a100" && ${device_type} != "h100" && ${device_type} != "4090") ]]; then
        log "Invalid device_type: ${device_type}"
        print_help
        exit 1
    fi

    if [[ -z ${device_id} || ! ${device_id} =~ ^[0-9]+$ ]]; then
        log "Invalid device_id: ${device_id}"
        print_help
        exit 1
    fi

    if [[ ${num_procs} -eq 0 ]]; then
        log "Need at least 1 model configuration to run"
        print_help
        exit 1
    fi

    if [[ ${distribution} != "poisson" && ${distribution} != "point" && ${distribution} != "closed" ]]; then
        log "Invalid distribution: ${distribution}"
        print_help
        exit 1
    fi

    if [[ ! -z ${min_freq} ]]; then
        remainder=$(( (1410 - min_freq) % 15 ))
        if [[ ${remainder} -ne 0 ]]; then
            log "min_freq must satisfy: (1410-min_freq) mod 15 = 0"
            log "Got min_freq=${min_freq}, remainder=${remainder}"
            print_help
            exit 1
        fi
    fi

    if [[ ! -z ${max_freq} ]]; then
        remainder=$(( (1410 - max_freq) % 15 ))
        if [[ ${remainder} -ne 0 ]]; then
            log "max_freq must satisfy: (1410-max_freq) mod 15 = 0"
            log "Got max_freq=${min_freq}, remainder=${remainder}"
            print_help
            exit 1
        fi
    fi

    for mode in ${modes[@]}
    do
        if [[ ${mode} != "mps-uncap" && ${mode} != "mps-equi" && ${mode} != "mps-miglike" && ${mode} != "mps-throttle" && ${mode} != "mps-manual-cap" && ${mode} != "mig" && ${mode} != "tm" && ${mode} != "mig-profile-3" && ${mode} != "mig-profile-2" && ${mode} != "mig-profile-1" && ${mode} != "custom-mig" ]]; then
            log "Invalid mode: ${mode}"
            log "Must be one of: mps-uncap, mps-equi, mps-miglike, mig, ts, orion, mig-profile-3, mig-profile-2, mig-profile-1"
            print_help
            exit 1
        fi
    done
}

parse_input() {
    pattern="^[^-]+-[^-]+-[^-]+-[^-]+-[^-]+-[^-]+-[^-]+-[^-]+$"
    model_types=()
    models=()
    models_and_batch_sizes=()
    batch_sizes=()
    mig_slices=()
    num_reqs=()
    load_starts=()
    load_ends=()
    load_steps=()
    rpss=()
    for (( i=0; i<${num_procs}; i++ ))
    do
        element=${model_run_params[$i]}
        if [[ ! "${element}" =~ $pattern ]]; then
            log "Expected: Model-BatchSize-MIGSlice-NumReqs-LoadStart-LoadEnd-LoadSteps-Rps"
            log "Got: ${element}"
            log "Example: gpt-8-3-9223372-0.1-0.4-0.1-100"
            exit 1
        fi
        models[$i]=$(echo $element | cut -d'-' -f1)
        batch_sizes[$i]=$(echo $element | cut -d'-' -f2)
        mig_slices[$i]=$(echo $element | cut -d'-' -f3)
        num_reqs[$i]=$(echo $element | cut -d'-' -f4)
        models_and_batch_sizes[$i]=${models[$i]}"-"${batch_sizes[$i]}
        load_starts[$i]=$(echo $element | cut -d'-' -f5)
        load_ends[$i]=$(echo $element | cut -d'-' -f6)
        load_steps[$i]=$(echo $element | cut -d'-' -f7)
        rpss[$i]=$(echo $element | cut -d'-' -f8)
    done

    # Ensure number of iterations across each model is the same, regardless of their individual loads
    len=${#load_ends[@]}
    expected_value=""
    for ((i = 0; i < len; i++)); do
        start=${load_starts[i]}
        end=${load_ends[i]}
        step=${load_steps[i]}

        if [[ "$step" == "0" || "$step" == "0.0" ]]; then
            echo "Error: step size cannot be zero"
            exit 1
        fi
	start_scaled=$(echo "$start * 4" | bc)
	end_scaled=$(echo "$end * 4" | bc)
	step_scaled=$(echo "$step * 4" | bc)

	start_scaled=${start_scaled%.*}
	end_scaled=${end_scaled%.*}
	step_scaled=${step_scaled%.*}

	value=$(( (end_scaled - start_scaled) / step_scaled ))
        # value=$(echo "scale=2; ($end - $start) / $step" | bc)
        
        if [[ -z "$expected_value" ]]; then
            expected_value=$value
        elif [[ "$value" -ne "$expected_value" ]]; then
            echo "Mismatch number of load iterations across models at index $i: expected $expected_value, got $value."
            # exit 1
        fi
    done

}

setup_expr() {
    source helper.sh && helper_setup
    WS=$(git rev-parse --show-toplevel)

    # Get result dir
    distribution_types=()
    for (( i=0; i<${num_procs}; i++ ))
    do
        distribution_types+=(${distribution})
    done

    get_result_dir models[@] batch_sizes[@] distribution_types[@] ${device_type}

    # Add trap for cleanup and prints
    trap print_log_location EXIT
    PRINT_OUTS=/tmp/print_outs-$(uuidgen | cut -c 1-8).txt
    rm -f ${PRINT_OUTS}
    echo "Logs at ${PRINT_OUTS}"

}

generate_json_input() {
    export model="\"$1\""
    export batch_size=$2
    export distribution_type="\"$3"\"
    export requests_per_second=$4
    export mig_slice=$5
    export num_req=$6

    template=$(cat model-param.json.template)
    json_input=$(echo $template | envsubst)

    unset model batch_size distribution_type requests_per_second mig_slice num_req rps
}

generate_model_params() {
    declare -a json_arr=("${!1}")
    model_params='[]'
    for json_str in "${json_arr[@]}"; do
        model_params=$(jq ". += [$json_str]" <<< "$model_params")
    done
}

run_cmd() {
    local cmd_arg="$1"
    local device_id_arg=$2
    local mode_arg=$3
    local duration_arg=$4
    local run_id_arg=$5

    echo "Running: ${cmd_arg}" >> ${PRINT_OUTS}
    eval "${cmd_arg} >> ${PRINT_OUTS} 2>&1 &"
    run_expr_pid=$!

    # Wait for the pipe to be created
    pipe=/tmp/${run_id_arg}
    local ctr=0
    while [[ ! -p ${pipe} && $ctr -lt 100 ]];
    do
        sleep 0.01
        ctr=$((ctr+1))
    done
    sleep 1

    # Start the experiment
    timeout 1 bash -c "echo '{\"device-id\": ${device_id_arg}, \"mode\": \"${mode_arg}\"}' > ${pipe}"
    # Wait for the experiment to start
    load_ctr=10000
    while [[ ${load_ctr} -gt 0 ]];
    do
        if [[ -f /tmp/${run_id_arg}.load ]]; then
            break
        fi

        if ! kill -0 "${run_expr_pid}" &> /dev/null; then
            echo "Process no longer alive"
            exit 1
        fi

        ((load_ctr--))
        sleep 0.25
    done
    echo "Made it before while loop for duration arg"
    echo "Duration is ${duration_arg}"

    # Wait either for duration or num_reqs to complete
    local ctr=0
    while :
    do
        sleep 1
        ctr=$((ctr+1))

        # Experiment to finish after specific duration
        if [[ ${duration_arg} -gt 0 ]]; then
            # Waiting for sleep duration
            if [[ ${ctr} -eq ${duration_arg} ]]; then
                break
            fi

            # If the experiment dies before that, error
            if ! taskset -c 0 kill -0 ${run_expr_pid} 2>/dev/null; then
                exit 1
            fi

        # Experiment to finish after num_reqs completed, wait for process to die
        elif [[ ${duration_arg} -eq 0 ]]; then
            if ! taskset -c 0 kill -0 ${run_expr_pid} 2>/dev/null; then
                break
            fi
        fi
    done

    # Stop the experiment
    timeout 1 bash -c "echo '{\"mode\": \"stop\"}' > ${pipe}"

    # Wait for the process to exit
    wait ${run_expr_pid}
    run_expr_exit_code=$?
    if [[ ${run_expr_exit_code} -ne 0 ]]; then
        exit ${run_expt_exit_code}
    fi
    echo -e "===============================\n\n" >> ${PRINT_OUTS}

}

generate_closed_loop_load() {
    json_array=()
    for (( i=0; i<${num_procs}; i++ ))
    do
        generate_json_input ${models[$i]} ${batch_sizes[$i]} "closed" 0 ${mig_slices[$i]} ${num_reqs[$i]}
        json_array+=("${json_input}")
    done

    generate_model_params json_array[@]
    for mode in ${modes[@]}
    do
        local run_id_arg=$(uuidgen)
        cmd="./run_job_mix.sh \
            --device-type ${device_type} \
            --load 1 \
            --run-id ${run_id_arg} \
            --min-freq ${min_freq} \
            --max-freq ${max_freq} \
            '${model_params}'"
        log "Running closed loop experiment for ${mode}"
        run_cmd "${cmd}" ${device_id} ${mode} ${duration} ${run_id_arg}

        # Maybe compute stats here...
    done
}

generate_distribution_load() {
    
    # Iteratve over each GPU sharing mechanism
    for mode in ${modes[@]}
    do
        num_loads=$(echo "(${load_ends[0]} - ${load_starts[0]}) / ${load_steps[0]} + 1" | bc -l | awk '{print int($1)}')
        # Iterate over each laod setting
        for ((i = 0; i < num_loads; i++)); do
            
            # Prep each processes parameters
            json_array=()
            for ((j=0; j<${num_procs}; j++)); do
                step_increment=$(multiply_and_round ${load_steps[j]} $i 2)
                ratio=$(echo "${load_starts[j]} + $step_increment" | bc)
                rps=$(multiply_and_round $ratio ${rpss[j]})

                generate_json_input ${models[$j]} ${batch_sizes[$j]} ${distribution} ${rps} ${mig_slices[$j]} ${num_reqs[$j]}
                json_array+=("${json_input}")
            done

            # Get parameters for the model
            generate_model_params json_array[@]

            # Run the command
            local run_id_arg=$(uuidgen)
            cmd="./run_job_mix.sh \
                --device-type ${device_type} \
                --run-id ${run_id_arg} \
                --min-freq ${min_freq} \
                --max-freq ${max_freq} \
                '${model_params}'"
        log "Running ${distribution} experiment for ${mode} with round #${i}"
        run_cmd "${cmd}" ${device_id} ${mode} ${duration} ${run_id_arg}
        done
    done
}

get_input $@
validate_input
parse_input
setup_expr
if [[ ${distribution} == "closed" ]]; then
    # Generate closed loop load
    generate_closed_loop_load
else
    # Generate distribution load
    generate_distribution_load
fi
log "Run success: results are stored in ${result_dir}"



