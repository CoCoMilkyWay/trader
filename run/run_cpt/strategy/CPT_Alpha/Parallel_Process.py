import os
import time
import ctypes
import psutil
from multiprocessing import Process, sharedctypes, Lock
from typing import Dict
from viztracer import VizTracer, get_tracer

# -------- Configurable Constants --------
MAX_CODE_SIZE = 64  # Fixed size for 'code' field to avoid dynamic memory allocation
RING_BUFFER_SIZE = 256  # Ring buffer for signaling worker->main
CPU_BACKOFF = 0.0001

DATA_EMPTY = 0
DATA_INPUT_READY = 1
DATA_OUTPUT_READY = 2

CONTROL_CLEAR = 0
CONTROL_INITED = 1
CONTROL_STOP = 2
# -------- Shared Memory Structures --------

class SharedData(ctypes.Structure):
    """Structure for storing input/output data in shared memory."""
    _fields_ = [
        # Status flag
        ('status', ctypes.c_int),  # EMPTY, INPUT_READY, OUTPUT_READY
        
        # Input fields
        ('code', ctypes.c_char * MAX_CODE_SIZE),
        ('date', ctypes.c_uint),
        ('time', ctypes.c_uint),
        ('open', ctypes.c_float),
        ('high', ctypes.c_float),
        ('low', ctypes.c_float),
        ('close', ctypes.c_float),
        ('vol', ctypes.c_float),
        
        # Output fields
        ('signal', ctypes.c_int),
        ('value', ctypes.c_float),
    ]
    
class SharedControl(ctypes.Structure):
    """Structure for flow control."""
    _fields_ = [
        ('init', ctypes.c_int * 256), # 256 workers max
        ('stop', ctypes.c_int),
    ]
    
class SharedRingBuffer(ctypes.Structure):
    """Structure representing a finite ring buffer."""
    _fields_ = [
        ('head', ctypes.c_int),    # The next slot to be written to
        ('tail', ctypes.c_int),    # The next slot to be read from
        ('buffer', ctypes.c_int * RING_BUFFER_SIZE)  # Stores indices of processed memory slots
    ]

# -------- Utility Functions --------

def set_cpu_affinity(cpu_id: int):
    """Pin the current process to a specific CPU for better CPU utilization."""
    # linux: os.sched_setaffinity(0, {cpu_id})
    p = psutil.Process()
    p.cpu_affinity([cpu_id])

# -------- Main ParallelProcess Class --------

class Parallel_Process:
    def __init__(self, code_info: Dict[str, Dict], Process_Core, Parallel:bool=False):
        """Initialize workers, shared memory, and setup shared metadata."""
        self.code_info = code_info          # Metadata about codes → determines worker memory allocation
        self.Process_Core = Process_Core    # Process_Core class
        self.Parallel = Parallel            # Whether use parallel at all
        self.num_workers = 0                # Number of workers
        self.num_codes_per_worker = []      # Number of codes handled by each worker
        self.worker_codes = []              # Codes assigned to each worker
        self.shared_mem = []                # Array of shared memory for each worker
        self.ring_buffers = []              # Ring buffers for signaling results
        self.locks = []                     # Locks for accessing ring buffers
        self.workers = []                   # Worker processes
        
        if not self.Parallel:
            self.C = Process_Core(0, [key for key in self.code_info.keys()])
            return
        
        logical_cpus = psutil.cpu_count(logical=True)
        physical_cpus = psutil.cpu_count(logical=False)
        if logical_cpus is None or physical_cpus is None:
            raise Exception('Could not determine CPU count')
        
        # ignore hyper-thread of intel cpu(we want more cache hit)
        hyper_thread = int(logical_cpus / physical_cpus)
        cpu = physical_cpus - 1
        
        # Pin main process to CPU 0
        set_cpu_affinity(0)
        
        # Determine the number of workers based on the available logical CPUs
        self.num_workers = min(cpu, physical_cpus-1, len(self.code_info))
        self.num_codes_per_worker = [0 for _ in range(self.num_workers)]
        self.worker_codes = [[] for _ in range(self.num_workers)]
        
        # Assign code to workers
        for code, info in self.code_info.items():
            idx = info['idx']
            worker_id = idx % self.num_workers
            mem_idx = idx // self.num_workers
            self.code_info[code]["worker_id"] = worker_id
            self.code_info[code]["mem_idx"] = mem_idx
            self.worker_codes[worker_id].append(code)
            self.num_codes_per_worker[worker_id] += 1

        # Initialize shared memory and workers
        print(f"Initializing: [Main] 1 process -> CPU0,")
        print(f"              [Worker] {self.num_workers} process -> {logical_cpus} Logical ({physical_cpus} Physical) CPUs: ...")
        self.shared_control = sharedctypes.RawValue(SharedControl)
        self.shared_control.init[:] = [CONTROL_CLEAR] * 256
        self.shared_control.stop = CONTROL_CLEAR
        for i in range(self.num_workers):
            shared_mem = sharedctypes.RawArray(SharedData, self.num_codes_per_worker[i])
            for j in range(self.num_codes_per_worker[i]):
                shared_mem[j].status = DATA_EMPTY
                
            ring_buffer = sharedctypes.RawValue(SharedRingBuffer)
            ring_buffer.head = 0
            ring_buffer.tail = 0
            
            lock = Lock()
            
            p = Process(
                target=self.parallel_exec,
                args=(i, self.worker_codes[i], shared_mem, self.shared_control, ring_buffer, lock, self.Process_Core),
                name=f"Worker-{i + 1}",
                daemon=True
            )
            p.start()
            
            self.shared_mem.append(shared_mem)
            self.ring_buffers.append(ring_buffer)
            self.locks.append(lock)
            self.workers.append(p)
            
    def check_workers(self):
        if not self.Parallel:
            return
        for i in range(self.num_workers):
            while True:
                initiated = self.shared_control.init[i] == CONTROL_INITED
                if initiated:
                    # print(f"Worker {i+1} signaled ready")
                    break
                time.sleep(CPU_BACKOFF)  # Yield CPU to avoid busy-waiting
                
        print("Parallel init complete.")
        
    def parallel_feed(self, code: str, new_bar: dict):
        """Feed new data to the appropriate worker based on the code."""
        if not self.Parallel:
            self.C.on_bar(code, new_bar['open'], new_bar['high'], new_bar['low'], new_bar['close'], new_bar['vol'], new_bar['time'])
            return
        
        worker_id = self.code_info[code]["worker_id"]
        mem_idx = self.code_info[code]["mem_idx"]
        shared_mem = self.shared_mem[worker_id]
        data = shared_mem[mem_idx]
        
        # print(f'Feeding worker {worker_id} mem {mem_idx}')
        
        status = data.status # this pass value, not pointer, thus safe
        if status == DATA_EMPTY:  # Only write to the slot if it's empty
            # Write the new data into shared memory
            data.code = code.encode('utf-8')
            data.date = new_bar['date']
            data.time = new_bar['time']
            data.open = new_bar['open']
            data.high = new_bar['high']
            data.low = new_bar['low']
            data.close = new_bar['close']
            data.vol = new_bar['vol']
            data.status = DATA_INPUT_READY
        elif status == DATA_INPUT_READY:
            raise RuntimeError(f"Error: Worker {worker_id} is still processing the previous input.")
        elif status == DATA_OUTPUT_READY:
            raise RuntimeError(f"Error: Worker {worker_id} has uncollected output.")
        
        # Notify the worker process via the ring buffer
        ring_buffer = self.ring_buffers[worker_id]
        lock = self.locks[worker_id]
        with lock:
            next_slot = (ring_buffer.head + 1) % RING_BUFFER_SIZE
            if next_slot != ring_buffer.tail:  # Ensure ring buffer isn't full
                ring_buffer.buffer[ring_buffer.head] = mem_idx
                ring_buffer.head = next_slot
            else:
                raise RuntimeError(f"Worker {worker_id}: Ring buffer is full, data {mem_idx} lost.")
        
    @staticmethod
    def parallel_exec(worker_id:int, worker_codes:list[str], shared_mem, shared_control, ring_buffer, lock, Process_Core):
        """
        Pin this worker to a dedicated CPU (e.g., worker_id + 1)
                    cpu0,     cpu1, cpu2, cpu3
                    {1 main}  { 3 workers    }
        worker_id:  NaN       0     1     2
        """
        
        set_cpu_affinity((worker_id+1)*2)  # Pin each worker to a specific core
        print(f'Worker Initiated...')
        
        C = Process_Core(worker_id, worker_codes)
        
        shared_control.init[worker_id] = CONTROL_INITED
        
        while shared_control.stop != CONTROL_STOP: # Keep processing indefinitely unless terminated externally
            if ring_buffer.tail != ring_buffer.head:
                with lock:
                    mem_idx = ring_buffer.buffer[ring_buffer.tail]
                    ring_buffer.tail = (ring_buffer.tail + 1) % RING_BUFFER_SIZE
                    
                data = shared_mem[mem_idx]
                status = data.status
                if status == DATA_INPUT_READY:
                    # Process the data
                    # print(f'Processing worker {worker_id} mem {mem_idx} status {status}')
                    C.on_bar(data.code.decode('utf-8'), data.open, data.high, data.low, data.close, data.vol, data.time)
                    data.status = DATA_OUTPUT_READY
                elif status == DATA_OUTPUT_READY:
                    raise Exception(f'Err: worker ({worker_id}) fed repeatedly')
                elif status == DATA_EMPTY:
                    # waiting to be fed/processed in next query
                    pass
                else:
                    raise Exception(f"Err: Worker {worker_id} has unexpected status {status} at index {mem_idx}.")
                
            time.sleep(CPU_BACKOFF)  # Yield CPU to avoid busy-waiting
        
        C.on_backtest_end()
    
    def parallel_collect(self):
        """Retrieve processed results from a specific worker."""
        
        if not self.Parallel:
            return
        
        # print('Start collecting')
        results = []
        for w in range(self.num_workers):
            shared_mem = self.shared_mem[w]
            num_codes = self.num_codes_per_worker[w]
            for i in range(num_codes):
                while self.shared_control.stop != CONTROL_STOP:
                    data = shared_mem[i]
                    status = data.status
                    if status == DATA_OUTPUT_READY:
                        # print(f'Collecting worker {w} idx {i}')
                        results.append((data.code.decode('utf-8'), data.signal, data.value))
                        data.status = DATA_EMPTY  # Mark slot as empty for reuse
                        break
                    time.sleep(CPU_BACKOFF)  # Yield CPU to avoid busy-waiting
                    
        # print('End collecting')
        return results
    
    def parallel_close(self):
        """Gracefully terminate all workers."""
        
        if not self.Parallel:
            self.C.on_backtest_end()
            return
        
        self.shared_control.stop = CONTROL_STOP
        for worker in self.workers:
            worker.terminate()
            worker.join()
