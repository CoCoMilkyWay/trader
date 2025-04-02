# Calculation Architecture

## Layer 1: Data Processing & Feature Construction (Applicable to both backtesting and live trading)

### Flow
- **Data Collection & Handling:** Aggregation, preprocessing, and database construction.
- **Data Integration & Cleaning:** Handling multiple data sources and frequencies (e.g., k-bar, order book, fundamentals, event-driven data).
- **Feature Engineering:** Computationally intensive indicator construction.

### Advantages
- Unified implementation for both backtesting and live trading, ensuring backtest integrity.

### Challenges
- **Computational Overhead:**
  - Requires cross-sectional calculations for thousands of assets on each bar.
  - While real-time trading execution takes only a few milliseconds (except for high-frequency trading), backtesting, mining, and parameter searching can be extremely slow.
- **Processing Constraints:**
  - Primarily serial execution with fine-grained synchronization at the bar/tick level.
  - Single-threaded CPU execution (multi-threading and hyper-threading disabled due to inefficiencies).
  - Multiprocessing overhead is unjustified for most cases.

### Output
- A dense tensor (stored as multiple sparse/compressed tensors) representing time-series and cross-sectional features.

### Hardware
- Single CPU thread with minimal parallelism (attempting further parallelization is impractical).

---

## Layer 2: Alpha Construction & Strategy Research (Applicable to both backtesting and live trading)

### Flow
- **Alpha Formulation:** Constructed from the static dense feature tensor generated in Layer 1.
- **Alpha Mining:** Deep Reinforcement Learning (DRL)-based exploration.
- **Financial Research:** Traditional alpha/CTA (Commodity Trading Advisor) research leveraging domain knowledge.
- **Machine Learning:** DL/ensemble methods to generate composite ("super") alphas.
- **Strategy Optimization:** Parameter tuning for enhanced performance.

### Key Considerations
- Each alpha should be expressible using a limited set of features and operators.

### Advantages
- **Drastic Speed Improvement:** Eliminates the need to rerun Layer 1 for each backtest iteration.
- **Real-Trading Efficiency:** Adds minor overhead but remains viable for minute-level trading.
- **Decoupled Execution:** Precomputed features allow fully asynchronous calculations, eliminating concerns about:
  - Discrepancies between backtesting and live trading.
  - Future information leakage.

### Output
- Alpha signals, composite (super) alphas, and trading signals.

### Hardware
- GPU clusters (finally, some real parallelism!).

---

## Fast Backtesting, Research, and Alpha Mining

### GPU Computation Considerations
- **Dense tensors are required** for full compatibility with GPU operations, unlike sparse tensors.
- **Memory requirements for dense tensor storage:**
  - **20 years, 1-min data:** 5000 assets × 200 features × 16-bit precision → **~20 TiB**
  - **20 years, 5-min data:** 5000 assets × 200 features × 16-bit precision → **~582 GiB**
  - **1 year, 5-min data:** 5000 assets × 100 features × 16-bit precision → **~14 GiB** (maximum practical GPU batch size,  1~5 seconds to load each time).

### Tick Data Processing
- Requires multiple GPUs with daily batch execution for scalability.

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