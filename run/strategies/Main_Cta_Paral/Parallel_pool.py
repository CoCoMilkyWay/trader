from __future__ import (absolute_import, division, print_function, unicode_literals)
import time
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import List, Dict
from strategies.Main_Cta_Paral.Define import MetadataIn, MetadataOut
from strategies.Main_Cta_Paral.Processor import n_Processor

import signal
terminate = False
def signal_handler(signum, frame):
    global terminate
    terminate = True
    # print("Interrupt received, terminating...")
signal.signal(signal.SIGINT, signal_handler)

@staticmethod
def lock_print(lock, msg):
    with lock:
        print(msg)

import multiprocessing
import threading
class n_processor_queue:
    # ProcessPoolExecutor: CPU-bound tasks
    # ThreadPoolExecutor: I/O-bound tasks
    # slaves: processing many tasks independent of each other
    # master: processing slave products, have exclusive access to certain system resources
    def __init__(self, max_workers=1, concurrency_mode='process'):
        self.manager = multiprocessing.Manager()
        self.max_workers = min(multiprocessing.cpu_count(), max_workers)
        self.concurrency_mode = self.parallel_mode(concurrency_mode)
        
        # multiprocessing.Queue() has less inter-process overhead than manager.list()
        # self.in_queues = [self.manager.list() for _ in range(self.max_workers)]  # Shared list to store results
        # self.out_queues = [self.manager.list() for _ in range(self.max_workers)]  # Shared list to store results
        self.in_queues = [multiprocessing.Queue() for _ in range(self.max_workers)]  # Shared list to store results
        self.out_queues = [multiprocessing.Queue() for _ in range(self.max_workers)]  # Shared list to store results
        # self.print_lock = threading.Lock() if self.concurrency_mode == 'thread' else multiprocessing.Lock()
        # self.active_workers = multiprocessing.Value('i', 0)
        self.onfly_tasks = [multiprocessing.Value('i', 0) for _ in range(self.max_workers)]
        self.end = multiprocessing.Value('i', False)  # Shared memory for boolean
        self.workers_active = [multiprocessing.Event() for _ in range(self.max_workers)]  # Event for signaling
        self.workers = []
        self.is_running = False
        
    # N-gets corresponds to N-puts
    def add_in_task(self, tasks:List[MetadataIn]):
        for i, task in enumerate(tasks):
            self.in_queues[i % self.max_workers].put(task) # round-robin like
            self.onfly_tasks[i % self.max_workers].value += 1
        # self.active_workers.value = self.max_workers
        for i in range(self.max_workers):
            self.workers_active[i].set() # start worker
            
    # def add_out_task(self, worker_id, tasks):
    #     for task in tasks:
    #         self.out_queues[worker_id].put(task)

    # mutable types (like lists, dictionaries, and other objects)
    # can be modified in place (like pointers)
    @staticmethod # use global method or static method in a class
    def worker(
                in_queue:multiprocessing.Queue,
                out_queue:multiprocessing.Queue,
                # print_lock,
                # active_workers,
                onfly_tasks,
                # tqdm_cnt,
                # tqdm_desc,
                worker_id,
                worker_active,
                end, # type: ignore
               ):
        n_processor = n_Processor()
        global terminate
        while not terminate and not end.value: # if not sleep for too long, will be killed
            
            worker_active.wait()  # Blocks until the event is set
            
            if end.value:
                break
            
            tasks = []
            try:
                while onfly_tasks.value != 0:
                    tasks.append(in_queue.get(block=True, timeout=1))
                    onfly_tasks.value -= 1
                    
                results = n_processor.process_slave_task(worker_id, tasks)
                if results:
                    out_queue.put(results)  # Store results
                    
            except Exception as e:
                print(f"{type(e).__name__}")
                print(e)
                # Get the traceback information
                import traceback
                tb_str = traceback.format_exc()
                print(tb_str)
                break
            
            worker_active.clear()  # Reset the event to pause the worker
            
        n_processor.on_backtest_end()
        
    def parallel_mode(self, concurrency_mode):
        return concurrency_mode
    
    def start_workers(self):
    # for multiple processes: shared info (IO/function/class) needs to be 'pickled' first
    # use global function in worker process (easier to serialize than class)
    # use @staticmethod to mark function if is in a class
        self.workers = [] # thread or process
        for i in range(self.max_workers):
            w = multiprocessing.Process(target=self.worker, args=(
                # shareable:
                self.in_queues[i],
                self.out_queues[i],
                # self.print_lock,
                self.onfly_tasks[i],
                # self.worker_active[i],
                # self.tqdm_cnt_with_lock,
                # self.tqdm_desc,
                
                # non-shareable(inout):
                i,
                self.workers_active[i],
                self.end,
            )) if self.concurrency_mode == 'process' else threading.Thread(target=self.worker, args=(
                self.in_queues[i],
                self.out_queues[i],
                # self.print_lock,
                self.onfly_tasks[i],
                # self.worker_active[i],
                i,
                self.workers_active[i],
                self.end,
            ))
            w.start()
            self.workers.append(w)
        self.is_running = True
        # # Start a separate thread to update the progress bar
        # self.pbar_updater = threading.Thread(target=self.update_pbar)
        # self.pbar_updater.start()
        
    def stop_workers(self):
        self.end.value = 1
        for i in range(self.max_workers):
            self.workers_active[i].set() # start worker
        for w in self.workers:
            w.join()
        self.is_running = False
        # self.pbar_stop_event.set()
        # if hasattr(self, 'pbar_updater'):
        #     self.pbar_updater.join()
        print("All workers have been stopped.")
        
    def step_execute(self) -> List[MetadataOut]:
        if not self.is_running:
            self.start_workers()
            
        # Process current tasks in queues
        global terminate
        while not terminate:
            if all(not worker_active.is_set() for worker_active in self.workers_active):
                break
            # print(slaves_queue_idle,master_queue_idle,worker_idle )
            
        # Check if termination was requested
        if terminate:
            self.stop_workers()
            
        results:List[MetadataOut] = []
        for out_queue in self.out_queues:
            while not out_queue.empty():
                results.extend(out_queue.get())
        return results
    
    def terminate(self):
        self.stop_workers()
        