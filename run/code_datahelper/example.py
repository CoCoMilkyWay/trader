import multiprocessing
import time

# Function for the worker processes to execute
def worker(input_queue, output_queue):
    while True:
        task = input_queue.get()
        if task is None:
            # Stop the process when None is received
            break
        result = task**2  # Example task: squaring the number
        time.sleep(1)  # Simulate a time-consuming task
        output_queue.put(result)

if __name__ == "__main__":
    # Number of worker processes
    num_workers = 4

    # Create the queues
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Start the worker processes
    processes = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(task_queue, result_queue))
        p.start()
        processes.append(p)

    # Add tasks to the queue
    tasks = [1, 2, 3, 4, 5, 6, 7, 8]
    for task in tasks:
        task_queue.put(task)

    # Add None to the queue to signal the workers to stop
    for _ in range(num_workers):
        task_queue.put(None)

    # Collect the results
    results = []
    for _ in tasks:
        result = result_queue.get()
        results.append(result)

    # Ensure all processes have finished
    for p in processes:
        p.join()

    print("Results:", results)
