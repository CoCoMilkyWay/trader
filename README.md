# Calculation Architecture

## **Layer 1** (for both backtesting and trading)
### **Flow:**
- Data collection, handling, and database construction(not sql, use raw or binary(compressed) text file-based library)
- Diverse data types and cleaning:
  - different frequencies (aligned to a single time axis)
  - different sources: k-bar, order book, fundamental, event-driven, etc.
  - different construction method/models
  - different cleaning method
- Solving for data/feature calculation dependencies
- Indicator(feature) construction (extremely computation-intensive)

### **Pros:**
- Provides unified code for both real trading and backtesting, ensuring backtest integrity

### **Problems:**
- Heavy, slow, and complicated
- Requires cross-sectional calculations on each bar over thousands of assets
  - This is not an issue for real trading, as it takes only a few milliseconds (unless performing high-frequency trading)
  - However, it becomes extremely slow for backtesting, mining, and parameter searching
- Mostly serial and scalar, with bar/tick-level fine-grained synchronization:
  - Can only utilize CPU (single-threaded, you can disable hyper-threading and thread migration)
  - Most annoyingly, the cost of multi-processing is simply not justified

### **Output:**
- A huge dense tensor (stored as multiple sparse/compressed tensors) of time-series/cross-sectional features

### **Hardware:**
- Mostly a single CPU thread (with very limited parallelism, just sad :< )

---

## **Layer 2** (for both backtesting and trading)
### **Flow:**
- Constructs (through research/mining) alpha formulas/rules from the static dense feature tensor of Layer 1
- DRL (deep reinforcement learning) layer for alpha mining
- Standard alpha/CTA research utilizing financial domain knowledge
- ML (DL/ensemble) layer to construct super-alpha from the alpha pool
- Parameter tuning for strategies

### **Key:**
- Each alpha should be constructible using only a few features with a limited number of operators

### **Pros:**
- Completely eliminates the need to run Layer 1 for each backtest run, MASSIVELY increasing backtest speed
- Due to layer abstraction, it adds a slight overhead for real trading, but this is completely fine for minute-level trading
- Since all features are precomputed, fine-grained synchronization is no longer needed, allowing free computation without concerns about:
  - Backtest/real-trading discrepancies
  - Future information leaks

### **Output:**
- Alpha, super (composite) alpha, and trading signals

### **Hardware:**
- GPU clusters (happy now :>)

---

## **For Fast Backtesting, Research, and Alpha Mining:**
- For GPU-based calculations, only dense tensors support all operations, compared to sparse tensors.
- **Size calculations for dense tensors (time-series × cross-section × features):**
  - **20 years × 24 hrs × 1 min × 5000 assets × 200 features × 16b = 20 TiB**
  - **20 years × 5 hrs × 5 min × 5000 assets × 200 features × 16b = 582 GiB**
  - **1 year × 5 hrs × 5 min × 5000 assets × 100 features × 16b = 14 GiB** (this is the max batch size for GPU calculations)
- For tick data, multiple GPUs are required for batch runs on a daily basis(batch size).


- Sparse Tensor formats:
```
+-----------------------+----------------------------+----------------------------+----------------------------+-----------------------------------+
| Feature               | COO (Coordinate)           | CSR (Compressed Sparse Row)| CSC (Compressed Sparse Column)| LIL (List of Lists)            |
+-----------------------+----------------------------+----------------------------+----------------------------+-----------------------------------+
| Data Representation   | 3 arrays: row indices,     | 3 arrays: row pointers,    | 3 arrays: column pointers, | List of lists: (column, value)    |
|                       | column indices, values     | column indices, values     | row indices, values        | pairs for each row.               |
+-----------------------+----------------------------+----------------------------+----------------------------+-----------------------------------+
| Advantages            | Simple, intuitive,         | Great for row slicing,     | Great for column slicing,  | Flexible, good for dynamic        |
|                       | easy to construct, direct  | compact storage, fast      | preferred for column-based | matrix building, incremental      |
|                       | representation of nonzeros | matrix-vector multiplication| algorithms.               | updates.                          |
+-----------------------+----------------------------+----------------------------+----------------------------+-----------------------------------+
| Disadvantages         | No inherent order,         | Inefficient for column     | Slow row slicing, costly   | Memory inefficient, slow for      |
|                       | requires sorting,          | access, costly updates     | updates during modification| arithmetic operations.            |
|                       | duplicates need merging    | after assembly.            |                            |                                   |
+-----------------------+----------------------------+----------------------------+----------------------------+-----------------------------------+
| Insertion/Modification| Good for bulk insertion,   | Best for fixed matrices,   | Poor for dynamic insertion,| Extremely efficient for dynamic   |
|                       | not for frequent changes   | no frequent updates.       | batch construction.        | modifications, ideal for          |
|                       | after creation.            |                            |                            | constructing before converting.   |
+-----------------------+----------------------------+----------------------------+----------------------------+-----------------------------------+
| Arithmetic & Access   | Slower for arithmetic until| Fast for row-based         | Fast for column-based      | Slow for arithmetic, converted    |
| Performance           | sorted or converted.       | operations, matrix-vector  | operations, column-based   | to CSR/CSC for performance.       |
|                       |                            | multiplication.            | factorizations.            |                                   |
+-----------------------+----------------------------+----------------------------+----------------------------+-----------------------------------+
| Typical Use Cases     | Prototyping, intermediate  | Iterative solvers, finite  | Column-based solvers,      | Dynamic matrix construction,      |
|                       | format before conversion   | element methods, row-based | column slicing, factorizations| early stages before conversion.|
|                       | to CSR/CSC.                | operations.                |                            |                                   |
+-----------------------+----------------------------+----------------------------+----------------------------+-----------------------------------+
```