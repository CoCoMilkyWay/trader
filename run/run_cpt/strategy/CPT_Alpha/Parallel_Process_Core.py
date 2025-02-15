import time
import torch
import ctypes
import psutil
from multiprocessing import Process, sharedctypes, Lock
from typing import Dict

# -------- Configurable Constants --------
HYPER_THREAD = 2 # ignore hyper-thread of intel cpu(we want more cache hit)
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

class Parallel_Process_Core:
    def __init__(self, code_info: Dict[str, Dict], Process_Core, Parallel:bool=False):
        """Initialize workers, shared memory, and setup shared metadata."""
        self.code_info = code_info          # Metadata about codes → determines worker memory allocation
        self.num_timestamps = 0             # num_timestamps
        self.Process_Core = Process_Core    # Process_Core class
        self.Parallel = Parallel            # Whether use parallel at all
        self.num_workers = 0                # Number of workers
        self.worker_code_num = []           # Number of codes handled by each worker
        self.worker_code_info = []          # Codes assigned to each worker
        self.shared_data = []               # Array of shared memory for each worker
        self.ring_buffers = []              # Ring buffers for signaling results
        self.locks = []                     # Locks for accessing ring buffers
        self.workers = []                   # Worker processes
        
        # for meta data only
        self.C_dummy = Process_Core(-1, {'dummy_code':{'idx':-1}}, torch.zeros((1)))
        self.shared_tensor = self._init_shared_tensor()
        
        if not self.Parallel:
            self.C = Process_Core(0, self.code_info, self.shared_tensor)
            return
        
        logical_cpus = psutil.cpu_count(logical=True)
        physical_cpus = psutil.cpu_count(logical=False)
        if logical_cpus is None or physical_cpus is None:
            raise Exception('Could not determine CPU count')
        
        # hyper_thread = int(logical_cpus / physical_cpus)
        cpu = physical_cpus - 1
        
        # Pin main process to CPU 0
        set_cpu_affinity(0)
        
        # Determine the number of workers based on the available logical CPUs
        self.num_workers = min(cpu, physical_cpus-1, len(self.code_info))
        self.worker_code_info = [{} for _ in range(self.num_workers)]
        self.worker_code_num = [0 for _ in range(self.num_workers)]
        
        # Assign code to workers
        for code, info in self.code_info.items():
            idx = info['idx']
            worker_id = idx % self.num_workers
            code_idx = idx // self.num_workers
            self.code_info[code]["worker_id"] = worker_id
            self.code_info[code]["code_idx"] = code_idx
            self.worker_code_info[worker_id][code] = info
            self.worker_code_num[worker_id] += 1
            
        # Initialize shared memory and workers
        print(f"Initializing: [Main  ]  1 process -> Logical CPU0,")
        print(f"              [Worker] {self.num_workers:>2} process -> Logical CPU{[(i+1)*HYPER_THREAD for i in range(self.num_workers)]} (ignore hyper-thread for better cache hit)")
        self.shared_control = sharedctypes.RawValue(SharedControl)
        self.shared_control.init[:] = [CONTROL_CLEAR] * 256
        self.shared_control.stop = CONTROL_CLEAR
        for i in range(self.num_workers):
            shared_data = sharedctypes.RawArray(SharedData, self.worker_code_num[i])
            for j in range(self.worker_code_num[i]):
                shared_data[j].status = DATA_EMPTY
                
            ring_buffer = sharedctypes.RawValue(SharedRingBuffer)
            ring_buffer.head = 0
            ring_buffer.tail = 0
            
            lock = Lock()
            
            p = Process(
                target=self.parallel_exec,
                args=(i, self.worker_code_info[i], shared_data, self.shared_control, ring_buffer, lock, self.Process_Core, self.shared_tensor),
                name=f"Worker-{i + 1}",
                daemon=True
            )
            p.start()
            
            self.shared_data.append(shared_data)
            self.ring_buffers.append(ring_buffer)
            self.locks.append(lock)
            self.workers.append(p)
    
    def _init_shared_tensor(self):
        from config.cfg_cpt import cfg_cpt
        from Util.UtilCpt import time_diff_in_min
        self.start = cfg_cpt.start
        self.end = cfg_cpt.end
        N_timestamps = time_diff_in_min(self.start, self.end)
        N_features = len(self.C_dummy.feature_names)
        N_labels = len(self.C_dummy.label_names)
        N_columns = N_features + N_labels
        N_codes = len(self.code_info.keys())
        
        print(f"Initializing Pytorch Tensor: (timestamp({N_timestamps}), feature({N_features}) + label({N_labels}), codes({N_codes}))")
        # tensor(a,b,c) is stored like:
        #   for each value a, we have a matrix(b,c):
        #   for each value b, we have a vector(c): thus c are stored continuously in memory
        # Therefore, PyTorch uses a row-major (C-contiguous) memory layout
        # for more efficient cross-section data processing (how are features in each code compare), we store codes dimension together
        shared_tensor = torch.zeros((N_timestamps, N_columns, N_codes), dtype=torch.float16).share_memory_()
        return shared_tensor
    
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
        code_idx = self.code_info[code]["code_idx"]
        shared_data = self.shared_data[worker_id]
        data = shared_data[code_idx]
        
        # print(f'Feeding worker {worker_id} mem {code_idx}')
        
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
                ring_buffer.buffer[ring_buffer.head] = code_idx
                ring_buffer.head = next_slot
            else:
                raise RuntimeError(f"Worker {worker_id}: Ring buffer is full, data {code_idx} lost.")
        
    @staticmethod
    def parallel_exec(worker_id:int, worker_code_info:Dict[str, Dict], shared_data, shared_control, ring_buffer, lock, Process_Core, shared_tensor):
        """
        Pin this worker to a dedicated CPU (e.g., worker_id + 1)
                    cpu0,     cpu1, cpu2, cpu3
                    {1 main}  { 3 workers    }
        worker_id:  NaN       0     1     2
        """
        
        set_cpu_affinity((worker_id+1)*HYPER_THREAD)  # Pin each worker to a specific core
        # print(f'Worker Initiated...')
        
        C = Process_Core(worker_id, worker_code_info, shared_tensor)
        
        shared_control.init[worker_id] = CONTROL_INITED
        
        while shared_control.stop != CONTROL_STOP: # Keep processing indefinitely unless terminated externally
            if ring_buffer.tail != ring_buffer.head:
                with lock:
                    code_idx = ring_buffer.buffer[ring_buffer.tail]
                    ring_buffer.tail = (ring_buffer.tail + 1) % RING_BUFFER_SIZE
                    
                data = shared_data[code_idx]
                status = data.status
                if status == DATA_INPUT_READY:
                    # Process the data
                    # print(f'Processing worker {worker_id} mem {code_idx} status {status}')
                    C.on_bar(data.code.decode('utf-8'), data.open, data.high, data.low, data.close, data.vol, data.time)
                    data.status = DATA_OUTPUT_READY
                elif status == DATA_OUTPUT_READY:
                    raise Exception(f'Err: worker ({worker_id}) fed repeatedly')
                elif status == DATA_EMPTY:
                    # waiting to be fed/processed in next query
                    pass
                else:
                    raise Exception(f"Err: Worker {worker_id} has unexpected status {status} at index {code_idx}.")
                
            time.sleep(CPU_BACKOFF)  # Yield CPU to avoid busy-waiting
        
        C.on_backtest_end()
    
    def parallel_collect(self):
        """Retrieve processed results from a specific worker."""
        
        self.num_timestamps += 1
        
        if not self.Parallel:
            return
        
        # print('Start collecting')
        results = []
        for w in range(self.num_workers):
            shared_data = self.shared_data[w]
            num_codes = self.worker_code_num[w]
            for i in range(num_codes):
                while self.shared_control.stop != CONTROL_STOP:
                    data = shared_data[i]
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
        from Util.UtilCpt import mkdir
        meta = {
            'timestamps':[self.num_timestamps, self.start, self.end],
            'features':[],
            'labels':[],
            'codes': [],
        }
        for idx, feature in enumerate(self.C_dummy.feature_names):
            meta['features'].append((
                str(feature),
                str(self.C_dummy.feature_types[idx]),
                str(self.C_dummy.scaling_methods[idx]),
                ))
        for idx, label in enumerate(self.C_dummy.label_names):
            meta['labels'].append((str(label),))
        code_names = sorted(self.code_info.keys(), key=lambda k: self.code_info[k]['idx'])
        for idx, code in enumerate(code_names):
            meta['codes'].append((str(code),))
        mkdir('results/')
        torch.save(meta, './results/meta.pt')
        torch.save(self.shared_tensor, './results/tensor.pt')
        
        # torch.set_printoptions(profile="full")
        # print(self.shared_tensor)
        
        if not self.Parallel:
            self.C.on_backtest_end()
        
        for worker in self.workers:
            worker.terminate()
            worker.join()
