import os
import time
import psutil
import ctypes
import multiprocessing as mp
from multiprocessing import sharedctypes, Lock
from typing import Dict, Callable

# Combined structure for input, output and status
class SharedData(ctypes.Structure):
    _fields_ = [
        # Status flags
        ('status', ctypes.c_int),      # 0: empty, 1: input ready, 2: output ready
        
        # Input fields
        ('code'  , ctypes.c_char_p),   # variable length string
        ('date'  , ctypes.c_int   ),
        ('time'  , ctypes.c_int   ),
        ('open'  , ctypes.c_float ),
        ('high'  , ctypes.c_float ),
        ('low'   , ctypes.c_float ),
        ('close' , ctypes.c_float ),
        ('vol'   , ctypes.c_float ),
        
        # Output fields
        ('signal', ctypes.c_int   ),
        ('value' , ctypes.c_float ),
    ]

def set_cpu_affinity(cpu_id:int):
    # linux:
    # os.sched_setaffinity(0, {cpu_id})
    
    # windows:
    p = psutil.Process()
    p.cpu_affinity([cpu_id]) # logical_cpus

class Parallel:
    def parallel_init(self, code_info:Dict[str, Dict], parallel_worker:Callable):
        self.code_info = code_info
        
        logical_cpus = psutil.cpu_count(logical=True)
        physical_cpus = psutil.cpu_count(logical=False)
        if logical_cpus is None or physical_cpus is None:
            raise Exception('Could not determine CPU count')
        print(f'Logical CPUs: {logical_cpus}, Physical CPUs: {physical_cpus}')
        
        # Pin main thread to CPU 0.
        set_cpu_affinity(0)
        print('Setting main thread to CPU 0...')
        
        self.num_workers = logical_cpus - 1
        self.workers = [] # (process, shared_mem)
        self.pipes = [] # notify other CPU, enhance CPU efficiency, avoid busy waiting
        self.locks = [] # for thread safty
        self.pending_workers = set()  # Track workers with pending tasks
        
        for i in range(self.num_workers):
            # Single shared memory block per worker
            shared_mem = sharedctypes.RawArray(SharedData, 1)
            shared_mem[0].status = 0  # Initialize as empty
            lock = Lock()  # Create a lock per worker
            
            parent_conn, child_conn = mp.Pipe(duplex=True)
            
            try:
                p = mp.Process(
                    target=self.parallel_exec,
                    args=(i, shared_mem, child_conn, lock, parallel_worker),
                    name=f"Worker-{i+1}",
                    daemon=True, # Make process daemon so it exits when main process exits
                    )
                p.start()
                print(f"Process started with PID: {p.pid}")  # Debug print
                self.workers.append((p, shared_mem))
                self.pipes.append(parent_conn)
                self.locks.append(lock)
            except Exception as e:
                print(f"Error creating worker {i+1}: {e}")
                raise
        
        for i in range(self.num_workers):
            # spawning new shells for each worker
            # it could take very long time if you have conda etc.
            # do explicit confirmation
            parent_conn = self.pipes[i]
            if parent_conn.poll(timeout=None): # blocking until a pipe message is received
                msg = parent_conn.recv()
                if msg == 'init_ready':
                    print(f"Worker {i+1} signaled ready")
                else:
                    print(f"Unexpected message from worker {i+1}: {msg}")
                    p.terminate()
            else:
                print(f"Worker {i+1} failed to initialize (timeout)")
                p.terminate()

    def parallel_feed(self, code:str, period:str, newBar:dict):
        worker_idx = self.code_info[code]['idx'] % self.num_workers
        worker_id = worker_idx + 1 # 0 reserved for main thread
        print(code, self.code_info[code], worker_idx)
        shared_mem = self.workers[worker_idx][1]
        lock = self.locks[worker_idx]
        pipe = self.pipes[worker_idx]
        data = shared_mem[0]
        
        while True:
            # Check previous result
            if data.status == 2:  # Potentially output ready(shouldn't happen)
                print(f'Err: worker CPU({worker_id}) data not collected, discarding input...')
                pass

            if data.status == 1:  # Potentially input ready(shouldn't happen)
                print(f'Err: worker CPU({worker_id}) busy, discarding input...')
                pass

            # Write new data if slot is empty
            if data.status == 0: # Potentially empty
                with lock:
                    if data.status == 0:
                        # Write input data
                        data.code = code.encode('utf-8')
                        data.date = newBar['date']
                        data.time = newBar['time']
                        data.open = newBar['open']
                        data.high = newBar['high']
                        data.low = newBar['low']
                        data.close = newBar['close']
                        data.vol = newBar['vol']

                        data.status = 1  # mark as input ready
                pipe.send('new')  # notify worker
                break # return after exit lock
    
    @staticmethod
    def parallel_exec(worker_idx, shared_mem, conn, lock, parallel_worker:Callable):
        # Pin this worker to a dedicated CPU (e.g., worker_id + 1)
        #               cpu0,     cpu1, cpu2, cpu3
        #               {1 main}  { 3 workers    }
        # worker_id:    NaN       0     1     2
        worker_id = worker_idx + 1
        print(f"Worker {worker_id} inited")
        conn.send('init_ready')
        try:
            set_cpu_affinity(worker_id)
        except Exception as e:
            print(f"Worker {worker_id} could not set CPU affinity: {e}")

        data = shared_mem[0]
        while True:
            # Check if there is a message in the pipe (non-blocking check)
            if conn.poll():
                msg = conn.recv()
                if msg == 'new':
                    print(f"Worker {worker_id} received new data")
                    with lock: # feel free to lock longer
                        if data.status == 1:
                            # parallel_worker(worker_id, data)
                            # signal, value = parallel_worker(worker_id, data)
                            print(f"Worker {worker_id} processed: {data.code.decode('utf-8')}, {data.date}, {data.time}")
                            # data.signal = signal
                            # data.value = value
                            data.status = 2 # mark as output ready
                    conn.send('done')
                elif msg == 'stop':
                    print(f'Worker {worker_id} received termination signal')
                    break
            time.sleep(0.001)  # prevent busy waiting

    def parallel_block(self, timeout=None):
        """Block until all workers are done processing."""
        start_time = time.time()
        results = {}  # Store results from workers
        print(f'Waiting for workers to complete...')
        
        while self.pending_workers:
            # Check each pending worker
            for worker_idx in list(self.pending_workers):
                pipe = self.pipes[worker_idx]
                if pipe.poll():
                    msg = pipe.recv()
                    if msg == 'done':
                        with self.locks[worker_idx]:
                            data = self.workers[worker_idx][1][0]
                            if data.status == 2:
                                results[worker_idx] = (data.signal, data.value)
                                data.status = 0
                            else:
                                print(f'Err: worker CPU({worker_idx}) data not ready, discarding output...')
                        self.pending_workers.discard(worker_idx)
            
            if timeout is not None and time.time() - start_time > timeout:
                print(f'Err: time out, pending CPU: {self.pending_workers}')
                return False, results
            
            time.sleep(0.001)
        
        print(f'All workers completed')
        return True, results

    def parallel_close(self):
        success, results = self.parallel_block(timeout=5)
        if not success:
            print("Warning: Some workers did not complete before shutdown")
            
        for conn in self.pipes:
            conn.send('stop')
        for p, _ in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
