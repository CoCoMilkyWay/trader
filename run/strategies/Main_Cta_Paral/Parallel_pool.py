from __future__ import (absolute_import, division, print_function, unicode_literals)
import time
import os, sys
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import List, Dict
from strategies.Main_Cta_Paral.Define import MetadataIn, MetadataOut
from strategies.Main_Cta_Paral.Processor import n_Processor

import threading
import multiprocessing
# Lock, Barrier, BoundedSemaphore, Condition, Event, RLock, Semaphore, SemLock

import signal
import keyboard
class HotkeyManager:
    # too many python libraries react to ctrl+c or KeyboardInterrupt,
    # it cannot be completely disabled,
    # and would yield unpredictable result(especially for multi-processing), 
    # use ctrl+4 instead
    
    # NOTE: handler function is a new thread, non-blocking(async)
    def __init__(self, hotkey='ctrl+4', handler_thread=None, remove_hot_key_on_exit:bool=True):
        self.hotkey = None
        self.hotkey_combo = hotkey
        self.remove_hot_key_on_exit = remove_hot_key_on_exit
        self.handler = handler_thread or self.default_handler
        self.running = True
        
    def __enter__(self):
        # Register the hotkey with provided handler
        self.hotkey = keyboard.add_hotkey(self.hotkey_combo, self.handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hotkey and self.remove_hot_key_on_exit:
            keyboard.remove_hotkey(self.hotkey)
    
    def default_handler(self):
        print(f"{self.hotkey_combo} pressed! Using default handler...")
        self.running = False
        
@staticmethod
def lock_print(lock, msg):
    with lock:
        print(msg)

# NOTE: process synchronization in python is a big problem
#   1. use mutex lock to protect all other syncing element like: value, queue
#   2. queues(even with block=True), only offers sync to process itself, 
#       no sync guarantee to the other process (:<)
#   3. in practice, using mp.value to indicate whether a queue is synced works just fine,
#       however, it may also fail, be ware
#   4. if in some systems, mp.value syncing somehow failed, 
#       you may have to give mp.value some time.sleep()
SLEEP = 0.00
WORKER_TRACE_PRINT = True
TIMEOUT = None

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
        self.in_queues = [multiprocessing.Queue() for _ in range(self.max_workers)]  # Shared list to store results
        self.out_queues = [multiprocessing.Queue() for _ in range(self.max_workers)]  # Shared list to store results
        
        self.print_lock = threading.Lock() if self.concurrency_mode == 'thread' else multiprocessing.Lock()
        
        # synchronization primitives
        self.task_locks = [multiprocessing.Lock() for _ in range(self.max_workers)]
        self.result_locks = [multiprocessing.Lock() for _ in range(self.max_workers)]
        self.task_counts = [multiprocessing.Value('i', 0) for _ in range(self.max_workers)]
        self.result_counts = [multiprocessing.Value('i', 0) for _ in range(self.max_workers)]
        self.worker_ready = [multiprocessing.Event() for _ in range(self.max_workers)]
        self.task_complete = [multiprocessing.Event() for _ in range(self.max_workers)]
        
        # self.debug = [multiprocessing.Value('i', 0) for _ in range(self.max_workers)]
        
        self.end = multiprocessing.Value('i', False)  # Shared memory for boolean
        
        self.workers = []
        
        # grace termination
        self.is_running = False
        self.is_stopping = False
        self.main_process_stopped = False
        self.worker_process_stopped = False
                
    # mutable types (like lists, dictionaries, and other objects)
    # can be modified in place (like pointers)
    @staticmethod # use global method or static method in a class
    def worker(
                in_queue:multiprocessing.Queue,
                out_queue:multiprocessing.Queue,
                task_lock,
                result_lock,
                task_count,
                result_count,
                worker_ready,
                task_complete,
                print_lock,
                end,
                # tqdm_cnt,
                # tqdm_desc,
                # debug,
                
                worker_id,
               ):
        def sigint_handler_worker():
            # with print_lock:
            #     print(f'worker {worker_id} exiting...')
            pass
            
        n_processor = n_Processor(lock=print_lock)
        
        with HotkeyManager(handler_thread=sigint_handler_worker):
            try:
                while True:
                    worker_ready.wait()  # Blocks until the event is set

                    if end.value:
                        break
                    
                    tasks = []

                    results = []
                    # use block operation when getting and putting to
                    # ensure consistency
                    with task_lock:
                        while task_count.value != 0:
                            tasks.append(in_queue.get(block=True, timeout=TIMEOUT)) # prepare tasks
                            task_count.value -= 1

                    results = n_processor.process_slave_task(worker_id, tasks)

                    if results:
                        with result_lock:
                            out_queue.put(results, block=True, timeout=TIMEOUT)  # Store results
                            result_count.value += len(results)
                            time.sleep(SLEEP)

                    # Signal task completion
                    task_complete.set()
                    worker_ready.clear() # Reset the event to pause the worker
                
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    return
                elif worker_id == 0 and WORKER_TRACE_PRINT:
                    with print_lock:
                        print(f'Worker {worker_id} exception: {str(e)}')
                        import traceback
                        print(traceback.format_exc())
                        return
                    
            n_processor.on_backtest_end()
        
    def parallel_mode(self, concurrency_mode):
        return concurrency_mode
    
    def start_workers(self):
    # for multiple processes: shared info (IO/function/class) needs to be 'pickled' first
    # use global function in worker process (easier to serialize than class)
    # use @staticmethod to mark function if is in a class
    
        def sigint_handler_main():
            
            # 1. signal stopping requirement
            self.is_stopping = True
            
            # 2. wait for main process to stop at particular location
            while not self.main_process_stopped:
                pass
            print(f'graceful termination...')
            
            # 3. wait for all worker process to stop
            for i in range(self.max_workers):
                self.task_complete[i].wait()
            
            # 4. exit(join) all worker process
            self.stop_workers()
            
            # 5. now allow main process to exit
            self.worker_process_stopped = True
            
            # 6. this handler thread exit automatically after return
            
        
        with HotkeyManager(handler_thread=sigint_handler_main, remove_hot_key_on_exit=False):
            self.workers = [] # thread or process

            for i in range(self.max_workers):
                worker_cls = multiprocessing.Process if self.concurrency_mode == 'process' else threading.Thread
                w = worker_cls(
                    target=self.worker,
                    args=(
                    # shareable:
                    self.in_queues[i],
                    self.out_queues[i],
                    self.task_locks[i],
                    self.result_locks[i],
                    self.task_counts[i],
                    self.result_counts[i],
                    self.worker_ready[i],
                    self.task_complete[i],
                    self.print_lock,
                    self.end,
                    # self.tqdm_cnt_with_lock,
                    # self.tqdm_desc,
                    # self.debug[i],

                    # non-shareable(inout):
                    i,
                    ), 
                    daemon=True,
                    )

                w.start()
                self.workers.append(w)

            self.is_running = True
            # # Start a separate thread to update the progress bar
            # self.pbar_updater = threading.Thread(target=self.update_pbar)
            # self.pbar_updater.start()
    
    def stop_workers(self):
        if not self.is_running:
            return
        self.end.value = True
        for i in range(self.max_workers):
            self.worker_ready[i].set() # start worker
        for w in self.workers:
            w.join()
        
        self.is_running = False
        # self.pbar_stop_event.set()
        # if hasattr(self, 'pbar_updater'):
        #     self.pbar_updater.join()
        print("All workers have been stopped.")
    
    # N-gets corresponds to N-puts
    def add_in_task(self, tasks:List[MetadataIn]):
        for i, task in enumerate(tasks):
            worker_id = i % self.max_workers
            with self.task_locks[worker_id]:
                self.in_queues[worker_id].put(task, block=True, timeout=TIMEOUT) # round-robin like
                self.task_counts[worker_id].value += 1
                
        # time.sleep(SLEEP)
        
        for i in range(self.max_workers):
            with self.task_locks[i]:
                if self.task_counts[i].value > 0:
                    self.worker_ready[i].set() # start worker
                else:
                    self.task_complete[i].set() # indicate task finished
                    
    # def add_out_task(self, worker_id, tasks):
    #     for task in tasks:
    #         self.out_queues[worker_id].put(task, block=True, timeout=TIMEOUT)
    
    def step_execute(self) -> List[MetadataOut]:
        if not self.is_running:
            self.start_workers()
        
        if self.is_stopping:
            self.main_process_stopped = True
            while not self.worker_process_stopped:
                pass
            
            # NOTE: may not be clean because we use _exit
            os._exit(1) # skip other exception handling
        
        # Collect results with synchronization
        results: List[MetadataOut] = []
        for i in range(self.max_workers):
            self.task_complete[i].wait()
            with self.result_locks[i]:
                while self.result_counts[i].value != 0: # self.out_queues[i].empty():
                    result = self.out_queues[i].get(block=True, timeout=TIMEOUT)
                    self.result_counts[i].value -= len(result)
                    results.extend(result)
                self.task_complete[i].clear()
                
        return results