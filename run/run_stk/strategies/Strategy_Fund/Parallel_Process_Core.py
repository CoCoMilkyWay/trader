import time
import torch
import ctypes
import psutil
from multiprocessing import Process, sharedctypes, Lock
from typing import Dict

from .Parallel_Process_Worker import Parallel_Process_Worker

# -------- Configurable Constants ----------
# ignore hyper-thread of intel/amd cpu(we want more cache hit)
HYPER_THREAD = 2
CPU_BACKOFF = 0.0001

MAX_WORKER = 256
MAX_CODES_PER_WORKER = 256

DATA_EMPTY = 0
DATA_INPUT_READY = 1
DATA_OUTPUT_READY = 2

CONTROL_CLEAR = 0
CONTROL_SLV_INITED = 1
CONTROL_MST_CONFIRMED = 1
CONTROL_STOP = 2

# -------- ITC/IPC Shared Memory Structures --------


class SharedData(ctypes.Structure):
    """
    Structure for storing input/output data in shared memory.
    instance = num_worker * num_codes_per_worker
    """
    _fields_ = [
        # Status flag
        ('status', ctypes.c_int),  # EMPTY, INPUT_READY, OUTPUT_READY

        # Input fields(cross-section)
        ('cs_signal', ctypes.c_int),
        ('cs_value', ctypes.c_float),

        # Output fields(time-series)
        ('ts_signal', ctypes.c_int),
        ('ts_value', ctypes.c_float),
    ]


class SharedControl(ctypes.Structure):
    """
    Structure for flow control.
    instance = num_worker
    """
    _fields_ = [
        ('init', ctypes.c_int * MAX_WORKER),
        ('stop', ctypes.c_int),
    ]


class SharedRingBuffer(ctypes.Structure):
    """
    Structure representing a finite ring buffer.
    instance = num_worker
    this is just a notification pipe to minimize scanning cost over many small shared buffers
    """
    _fields_ = [
        ('head', ctypes.c_int),    # The next slot to be written to
        ('tail', ctypes.c_int),    # The next slot to be read from
        # Stores indices of processed memory slots
        ('buffer', ctypes.c_int * MAX_CODES_PER_WORKER)
    ]

# -------- Utility Functions --------


def set_cpu_affinity(cpu_id: int):
    """Pin the current process to a specific CPU for better CPU utilization."""
    # linux: os.sched_setaffinity(0, {cpu_id})
    p = psutil.Process()
    p.cpu_affinity([cpu_id])

# -------- Main ParallelProcess Class --------


class Parallel_Process_Core:
    def __init__(self, code_info: Dict[str, Dict]):
        """Initialize workers, shared memory, and setup shared metadata."""
        self.code_info = code_info           # Metadata about codes → determines worker memory allocation
        self.num_timestamps = 0              # num_timestamps
        self.num_workers = 0                 # Number of workers
        self.worker_code_num = []            # Number of codes handled by each worker
        self.worker_code_info = []           # Codes assigned to each worker
        self.shared_data = []                # Array of shared memory for each worker
        self.ring_buffers = []              # Ring buffers for signaling results
        self.locks = []                     # Locks for accessing ring buffers
        self.workers = []                    # Worker processes

        # get config data from dummy class
        self.C_dummy = Parallel_Process_Worker(
            -1, {'dummy_code': {'idx': -1}}, torch.zeros((1)), True)
        self.shared_tensor = self._init_shared_tensor()

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
        print(
            f"Initializing: [Main  (Cross-Section)]  1 process -> Logical CPU0,")
        print(
            f"              [Worker(Time-Series  )] {self.num_workers:>2} process -> Logical CPU{[(i+1)*HYPER_THREAD for i in range(self.num_workers)]} (disable hyper-threading)")
        self.shared_control = sharedctypes.RawValue(SharedControl)
        self.shared_control.init[:] = [CONTROL_CLEAR] * MAX_WORKER
        self.shared_control.stop = CONTROL_CLEAR
        for i in range(self.num_workers):
            assert self.worker_code_num[i] < MAX_CODES_PER_WORKER
            shared_data = sharedctypes.RawArray(
                SharedData, self.worker_code_num[i])
            for j in range(self.worker_code_num[i]):
                shared_data[j].status = DATA_EMPTY

            ring_buffer = sharedctypes.RawValue(SharedRingBuffer)
            ring_buffer.head = 0
            ring_buffer.tail = 0

            lock = Lock()

            # on Window, only 'spawn'(safer) is valid, compared to 'fork'(faster)
            p = Process(
                target=self.slave_process,
                args=(i, self.worker_code_info[i], shared_data, self.shared_control,
                      ring_buffer, lock, Parallel_Process_Worker, self.shared_tensor),
                name=f"Worker-{i + 1}",
                daemon=True
            )
            p.start()

            self.shared_data.append(shared_data)
            self.ring_buffers.append(ring_buffer)
            self.locks.append(lock)
            self.workers.append(p)

        # start master process
        self.master_process()

    def _init_shared_tensor(self):
        from config.cfg_stk import cfg_stk
        from Util.UtilStk import time_diff_in_min
        self.start = cfg_stk.start
        self.end = cfg_stk.end
        N_timestamps = int(time_diff_in_min(self.start, self.end)
                           * cfg_stk.max_trade_session_ratio)
        N_features = len(self.C_dummy.feature_names)
        N_labels = len(self.C_dummy.label_names)
        N_columns = N_features + N_labels
        N_codes = len(self.code_info.keys())

        print(
            f"Initializing Pytorch Tensor: (timestamp({N_timestamps}), feature({N_features}) + label({N_labels}), codes({N_codes}))")
        # float16 = 2 Bytes
        print(
            f"Memory reserving: {(N_timestamps * N_columns * N_codes)*2/(1024**2):.2f} MB")
        # tensor(a,b,c) is stored like:
        #   for each value a, we have a matrix(b,c):
        #   for each value b, we have a vector(c): thus c are stored continuously in memory
        # Therefore, PyTorch uses a row-major (C-contiguous) memory layout
        # for more efficient cross-section data processing (how are features in each code compare), we store codes dimension together
        shared_tensor = torch.zeros(
            (N_timestamps, N_columns, N_codes), dtype=torch.float16).share_memory_()
        return shared_tensor

    def master_process(self):
        # check slaves ready
        for i in range(self.num_workers):
            while True:
                initiated = self.shared_control.init[i] == CONTROL_SLV_INITED
                if initiated:
                    # print(f"Worker {i+1} signaled ready")
                    break
                time.sleep(CPU_BACKOFF)  # Yield CPU to avoid busy-waiting

        print("Parallel init complete.")

        for i in range(self.num_workers):
            self.shared_control.init[i] = CONTROL_MST_CONFIRMED

        while True:
            # send info =======================================================
            for code in self.code_info.keys():
                worker_id = self.code_info[code]["worker_id"]
                code_idx = self.code_info[code]["code_idx"]
                shared_data = self.shared_data[worker_id]
                data = shared_data[code_idx]

                # print(f'Feeding worker {worker_id} mem {code_idx}')
                status = data.status  # this pass value, not pointer, thus safe
                if status == DATA_EMPTY:  # Only write to the slot if it's empty
                    # Write the new data into shared memory
                    data.cs_signal = code_idx
                    data.cs_value = 0.0
                    data.status = DATA_INPUT_READY
                elif status == DATA_INPUT_READY:
                    raise RuntimeError(
                        f"Error: Worker {worker_id} is still processing the previous input.")
                elif status == DATA_OUTPUT_READY:
                    raise RuntimeError(
                        f"Error: Worker {worker_id} has uncollected output.")

                # Notify the worker process via the ring buffer
                ring_buffer = self.ring_buffers[worker_id]
                lock = self.locks[worker_id]
                with lock:
                    next_slot = (ring_buffer.head + 1) % MAX_CODES_PER_WORKER
                    if next_slot != ring_buffer.tail:  # Ensure ring buffer isn't full
                        ring_buffer.buffer[ring_buffer.head] = code_idx
                        ring_buffer.head = next_slot
                    else:
                        raise RuntimeError(
                            f"Worker {worker_id}: Ring buffer is full, data {code_idx} lost.")

            # receive info ====================================================
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
                            results.append((data.ts_signal, data.ts_value))
                            data.status = DATA_EMPTY  # Mark slot as empty for reuse
                            break
                        # Yield CPU to avoid busy-waiting
                        time.sleep(CPU_BACKOFF)

            # =================================================================
            self.num_timestamps += 1

    @staticmethod
    def slave_process(worker_id: int, worker_code_info: Dict[str, Dict], shared_data, shared_control, ring_buffer, lock, Process_Worker, shared_tensor: torch.Tensor):
        """
        Pin this worker to a dedicated CPU (e.g., worker_id + 1)
                    cpu0,     cpu1, cpu2, cpu3
                    {1 main}  { 3 workers    }
        worker_id:  NaN       0     1     2
        """

        # Pin each worker to a specific core
        set_cpu_affinity((worker_id+1)*HYPER_THREAD)

        C = Process_Worker(worker_id, worker_code_info, shared_tensor)
        shared_control.init[worker_id] = CONTROL_SLV_INITED

        print(f'Worker {worker_id} Initiated...')

        while True:
            if shared_control.init[worker_id] == CONTROL_MST_CONFIRMED:
                break
            time.sleep(CPU_BACKOFF)  # Yield CPU to avoid busy-waiting

        C.run()

        while shared_control.stop != CONTROL_STOP:  # Keep processing indefinitely unless terminated externally
            # use ring-buffer to save scanning cost
            if ring_buffer.tail != ring_buffer.head:
                with lock:
                    code_idx = ring_buffer.buffer[ring_buffer.tail]
                    ring_buffer.tail = (
                        ring_buffer.tail + 1) % MAX_CODES_PER_WORKER

                data = shared_data[code_idx]
                status = data.status
                if status == DATA_INPUT_READY:
                    # Process the data
                    # print(f'Processing worker {worker_id} mem {code_idx} status {status}')
                    cs_signal = data.cs_signal
                    cs_value = data.cs_value
                    data.ts_signal = code_idx
                    data.ts_value = 0.0
                    data.status = DATA_OUTPUT_READY
                elif status == DATA_OUTPUT_READY:
                    raise Exception(
                        f'Err: worker ({worker_id}) fed repeatedly')
                elif status == DATA_EMPTY:
                    # waiting to be fed/processed in next query
                    pass
                else:
                    raise Exception(
                        f"Err: Worker {worker_id} has unexpected status {status} at index {code_idx}.")

            time.sleep(CPU_BACKOFF)  # Yield CPU to avoid busy-waiting

        C.on_backtest_end()

    def parallel_close(self):
        """Gracefully terminate all workers."""
        from Util.UtilCpt import mkdir
        meta = {
            'timestamps': [self.num_timestamps, self.start, self.end],
            'features': [],
            'labels': [],
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
        code_names = sorted(self.code_info.keys(),
                            key=lambda k: self.code_info[k]['idx'])
        for idx, code in enumerate(code_names):
            meta['codes'].append((str(code),))
        mkdir('results/')
        torch.save(meta, './results/meta.pt')
        torch.save(self.shared_tensor, './results/tensor.pt')

        # torch.set_printoptions(profile="full")
        # print(self.shared_tensor)

        for worker in self.workers:
            worker.terminate()
            worker.join()
