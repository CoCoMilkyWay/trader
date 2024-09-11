from __future__ import (absolute_import, division, print_function, unicode_literals)
import time
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import signal
terminate = False
def signal_handler(signum, frame):
    global terminate
    terminate = True
    print("Interrupt received, terminating...")
signal.signal(signal.SIGINT, signal_handler)

import multiprocessing
import threading
from strategies.Main_Cta_Paral.Processor import SlaveProcessor, MasterProcessor
# class SlaveProcessor:
#     def __init__(self):
#         self.state = {}  # Initialize any state variables here
# 
#     def process_slave_task(self, id, task, meta):
#         # Use and update self.state as needed
#         result = []  # Process the task and produce a result
#         return result
# class MasterProcessor:
#     def __init__(self):
#         self.state = {}  # Initialize any state variables here
# 
#     def process_master_task(self, id, task, meta):
#         # Use and update self.state as needed
#         pass  # Process the task
class n_slave_1_master_queue:
    # ProcessPoolExecutor: CPU-bound tasks
    # ThreadPoolExecutor: I/O-bound tasks
    # slaves: processing many tasks independent of each other
    # master: processing slave products, have exclusive access to certain system resources
    def __init__(self, max_workers=1, concurrency_mode='process'):
        self.max_workers = min(multiprocessing.cpu_count(), max_workers)
        self.concurrency_mode = self.parallel_mode(concurrency_mode)
        self.slave_tasks_queues = [multiprocessing.Queue() for _ in range(self.max_workers)]
        self.master_task_queue = multiprocessing.Queue()
        self.master_lock = threading.Lock() if self.concurrency_mode == 'thread' else multiprocessing.Lock()
        # self.tqdm_cnt_with_lock = multiprocessing.Value('i', 0)  # 'i': signed integer
        # self.tqdm_desc = multiprocessing.Array('c', 256)  # 'c': char
        # self.pbar = tqdm(total=tqdm_total) # tqdm pbar is not 'pickleable', so create a shareable int
        # self.pbar_stop_event = threading.Event()
        
    # def update_pbar(self):
    #     while not self.pbar_stop_event.is_set():
    #         current_value = self.tqdm_cnt_with_lock.value
    #         current_desc = self.tqdm_desc.value.decode().rstrip('\x00') # type: ignore
    #         self.pbar.n = current_value
    #         self.pbar.set_description(f"Processing {current_desc}")
    #         self.pbar.refresh()
    #         if current_value >= self.pbar.total:
    #             break
    #         time.sleep(0.1)  # Update every 0.1 seconds
    
    # N-gets corresponds to N-puts
    def add_slave_task(self, tasks):
        for i, task in enumerate(tasks):
            self.slave_tasks_queues[i % self.max_workers].put(task)
    def add_master_task(self, tasks):
        for task in tasks:
            self.master_task_queue.put(task)
            
    # mutable types (like lists, dictionaries, and other objects)
    # can be modified in place (like pointers)
    @staticmethod # use global method or static method in a class
    def worker(
                slave_tasks_queue,
                master_tasks_queue,
                master_lock,
                # tqdm_cnt,
                # tqdm_desc,
                worker_id,
                meta,
               ):
        slave_processor = SlaveProcessor()
        master_processor = MasterProcessor()
        
        global terminate
        while not terminate:
            # print('worker starts')
            if not master_tasks_queue.empty() and not terminate:
                master_task = master_tasks_queue.get()
                time.sleep(0.1)  # Prevent busy waiting (only master tasks left)
                with master_lock:
                    # print('worker acquired lock')
                    # print('worker processing master task')
                    master_processor.process_master_task(worker_id, master_task, meta)
                    # print('worker released lock')
            if not slave_tasks_queue.empty() and not terminate:
                slave_task = slave_tasks_queue.get()
                # print('worker processing slave task')
                result = slave_processor.process_slave_task(worker_id, slave_task, meta)
                master_tasks_queue.put(result)
                # with tqdm_cnt.get_lock(): # avoid racing condition
                #     tqdm_cnt.value += 1
                #     tqdm_info = slave_task # list(slave_task.keys())[0]
                #     tqdm_desc.value = str(tqdm_info).encode()[:255]  # Ensure it fits in the array
            # if master_tasks_queue.empty() and slave_tasks_queue.empty():
            #     # Stop the process when None is received
            #     # print('worker finished processing')
            #     break
        # Add a small sleep to allow for interrupt checking
        time.sleep(0.1)
        
    def parallel_mode(self, concurrency_mode):
        return concurrency_mode
    
    def execute(self):
        # for multiple processes: shared info (IO/function/class) needs to be 'pickled' first
        # use global function in worker process (easier to serialize than class)
        # use @staticmethod to mark function if is in a class
        meta = 0
        workers = [] # thread or process
        global terminate
        try:
            if self.concurrency_mode == 'process':
                for i in range(self.max_workers):
                    w = multiprocessing.Process(
                        target=self.worker,
                        args=(
                            # shareable:
                            self.slave_tasks_queues[i],
                            self.master_task_queue,
                            self.master_lock,
                            # self.tqdm_cnt_with_lock,
                            # self.tqdm_desc,

                            # non-shareable(inout):
                            i,
                            meta,
                            ))
                    w.start()
                    workers.append(w)
            elif self.concurrency_mode == 'thread':
                for i in range(self.max_workers):
                    w = threading.Thread(
                        target=self.worker,
                        args=(
                            # shareable:
                            self.slave_tasks_queues[i],
                            self.master_task_queue,
                            self.master_lock,
                            # self.tqdm_cnt_with_lock,
                            # self.tqdm_desc,

                            # non-shareable:
                            i,
                            meta,
                            ))
                    w.start()
                    workers.append(w)
            else:
                raise ValueError("Invalid concurrency mode")
        
            # # Start a separate thread to update the progress bar
            # self.pbar_updater = threading.Thread(target=self.update_pbar)
            # self.pbar_updater.start()
            
            for w in workers:
                w.join()
                
        except Exception as e:
            print(f"Execution error: {e}")
        finally:
            # self.pbar_stop_event.set()
            # if hasattr(self, 'pbar_updater'):
            #     self.pbar_updater.join()
            print("Execution completed or terminated.")
            
    def terminate(self):
        global terminate
        terminate = True
            