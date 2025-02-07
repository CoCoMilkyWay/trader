import multiprocessing
import time
import os

def worker_function(worker_id, large_data):
    print(f"Worker {worker_id} starting... Process ID: {os.getpid()}")
    # Simulate heavy imports (e.g., importing libraries within the worker)
    import numpy as np
    import pandas as pd
    # Simulate large data processing
    # print(f"Worker {worker_id} is processing large data of size {len(large_data)}...")
    processed_data = [x * 2 for x in large_data[:1000]]  # Process part of the data
    # Simulate blocking work
    # time.sleep(2)  # Simulates time-consuming task
    # print(f"Worker {worker_id} finished! Process ID: {os.getpid()}")

def run():
    num_processes = 5  # Number of worker processes to spawn
    processes = []
    print("Testing process spawn and worker execution with heavy imports and a large data object:")
    # Simulate a large data set to share
    large_data = [i for i in range(10**7)]  # A list with 10 million elements
    start_time = time.time()
    
    # Start all worker processes
    for i in range(num_processes):
        process = multiprocessing.Process(target=worker_function, args=(i, large_data))
        processes.append(process)
        process.start()  # Start each worker immediately

    # Wait until all worker processes finish
    for process in processes:
        process.join()
    
    end_time = time.time()
    print(f"Spawned and executed {num_processes} workers in {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    run()