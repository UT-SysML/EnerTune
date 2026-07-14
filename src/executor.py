import os
import threading
import argparse
import signal
import time
import timeit
import sys
import numpy as np
import pickle
import torch
from queue import Queue
from queue import Empty as QueueEmptyException
from inference import get_inference_object
from utils import (
    DistributionType,
    log
)

WARMUP_REQS = 1 

class InferenceExecutor:
    def __init__(
        self, model_obj, distribution_type, rps, tid, mig_slice,
        num_reqs=sys.maxsize,
        duration=-1
    ):
        self._model_obj = model_obj
        self._num_reqs = num_reqs
        self._tid = tid
        self._rps = rps
        self._mig_slice = mig_slice
        self._pid = os.getpid()

        # Process synchronization mechanism
        self._start = False
        self._finish = False
        self._job_completed = False
        self._duration = duration

        if distribution_type == "closed":
            self._distribution_type = DistributionType.CLOSED
        elif distribution_type == "point":
            self._distribution_type = DistributionType.POINT
            self._sleep_time = [1 / rps]
        elif distribution_type == "poisson":
            self._distribution_type = DistributionType.POISSON
            self._sleep_time = np.random.exponential(
                scale=(1 / rps),
                size=10000
            )
        else:
            print(f"Unknown distribution type: {distribution_type}")
            print("Allowed values: closed, point, poisson")
            sys.exit(1)
    
    def _catch_to_start(self, signum, frame):
        self._start = True
    
    def _catch_to_end(self, signum, frame):
        self._finish = True
        self._job_completed = True

    def _retire_experiment(self):
        time.sleep(self._duration)
        self._finish = True
    
    def _indicate_ready(self):
        # Wait til user instructs to start via signal handler
        self.install_signal_handler()
        # Write to indicate readiness, thereby the user
        # can signal when to start the process
        log(f"/tmp/{self._pid}", "")
        while not self._start:
            pass
            
        # Setup retire time if needed
        if self._duration > 0:
            retire_exp_thread = threading.Thread(
                target=self._retire_experiment,
            )
            retire_exp_thread.daemon = True
            retire_exp_thread.start()
    
    def _return_infer_stats(self, infer_stats):
        infer_stats.insert(0, f"{self._model_obj.get_id()}")
        result = (self._tid, self._mig_slice, self._pid, infer_stats)
        # Might want to change the name of this file
        # to not be in tmp and specified by user for latter retrieval
        with open(f"/tmp/{self._pid}.pkl", "wb") as h:
            pickle.dump(result, h)
    
    def _enqueue_requests(self, queue, num_reqs, is_warmup):
        sleep_array_len = len(self._sleep_time)

        for i in range(num_reqs):
            queued_time = time.time()
            queue.put(queued_time)
            if self._finish:
                break
            
            curr = timeit.default_timer()
            time.sleep(
                max(0,
                    self._sleep_time[i % sleep_array_len] - 
                    (timeit.default_timer() - curr)
                )
            )
        
        # Before existing, dump all items of the queue
        # if not is_warmup:
        #     while not queue.empty():
        #         queue.get()
        #     self._job_completed = True
        
    def install_signal_handler(self):
        signal.signal(signal.SIGUSR1, self._catch_to_start)
        signal.signal(signal.SIGUSR2, self._catch_to_end)
    
    def _run_infer_executor(self, num_reqs, i=0, is_warmup=False):
        print(f"We in run infer")
        # Create a queue to enqueue requests (for non close-loop experiments)
        enqueue_thread = None
        if self._distribution_type != DistributionType.CLOSED:
            queue = Queue()
            enqueue_thread = threading.Thread(
                target=self._enqueue_requests,
                args=(queue, num_reqs, is_warmup)
            )
            enqueue_thread.start()
        
        completed = 0
        total_time_arr = []
        queued_time_arr = []

        process_start_time = time.time()
        print(f"i is {i}")
        print(f"num_reqs is {num_reqs}")
        while i < num_reqs:
            if self._job_completed:
                break
            try:
                if self._distribution_type != DistributionType.CLOSED:
                    queued_time = queue.get(block=True, timeout=1)
                else:
                    queued_time = time.time()
            except QueueEmptyException:
                continue
            
            start_time=time.time()
            completed += self._model_obj.infer()
            print(f"Completed {completed} images")
            end_time = time.time()

            total_time = end_time - queued_time
            total_time_arr.append(total_time)
            queueing_delay = start_time - queued_time
            queued_time_arr.append(queueing_delay)

            i += 1
        
        process_end_time = time.time()
        if enqueue_thread:
            enqueue_thread.join()
        return [
            process_start_time,
            process_end_time,
            self._rps,
            completed,
            completed / (process_end_time - process_start_time),
            total_time_arr,
            queued_time_arr
        ]

    def run(self):
        # Load model and transfer inputs
        self._model_obj.load_model()
        self._model_obj.load_data()

        # Warm up the model
        print("Warming up")
        self._run_infer_executor(WARMUP_REQS)
        reqs_completed = WARMUP_REQS

        # Ready for experiment
        self._indicate_ready()

        # Start experiment
        if torch.cuda.is_available():
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("start")
        print("Bout to start real shit")
        infer_stats = self._run_infer_executor(
            self._num_reqs + reqs_completed,
            i=reqs_completed,
            is_warmup=False
        )
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        
        # Give stats back to user
        self._return_infer_stats(infer_stats)

if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Required params
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--distribution-type", type=str, required=True)
    parser.add_argument("--rps", type=float, required=True)
    parser.add_argument("--tid", type=int, required=True)

    # Optional parmas
    parser.add_argument("--num-reqs", type=int, default=sys.maxsize)
    parser.add_argument("--mig-slice", type=int, default=1)
    parser.add_argument("--run-id", type=str, default="fake")
    parser.add_argument("--uuid", type=str, default="fake")

    opt, unused_args = parser.parse_known_args()

    # Create inferene object
    model_obj = get_inference_object(
        opt.model,
        opt.device_id,
        opt.batch_size
    )

    final_rps = opt.rps / opt.batch_size
    final_num_reqs = opt.num_reqs / opt.batch_size
    final_num_reqs = int(final_num_reqs)
    # Create executor object
    executor_obj = InferenceExecutor(
        model_obj,
        opt.distribution_type,
        final_rps,
        opt.tid,
        opt.mig_slice,
        num_reqs=final_num_reqs,
    )

    executor_obj.run()

    # Quick testing of inference object
    # model_obj.load_model()
    # model_obj.load_data()
    # size = model_obj.infer()
    # print(size)
