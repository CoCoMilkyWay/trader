# calculation architecture:
- **Layer 1**: (for both backtest and trading)
    - Flow:
        - data collection, handling, database construction
        - diverse data types and cleaning (different frequency, different source: k-bar, orderbook, fundamental, event-driven, etc.)
        - indicator construction (extremely computation intensive)
    - Good:
        - provide unified code for both real-trading and backtesting, ensure backtest integrity
    - Problem:
        - Heavy, slow, complicated
        - Need cross section calculation on each bar over thousands of assets, 
            this is not a problem for real trading, as only take few ms (unless you do high-freq),
            but would be extremely slow for backtesting/mining/parameter-searching
        - Mostly Serial, Scalar with bar/tick-level-fine-grained sync:
            can only use cpu, and most annoyingly single-threaded
            most annoyingly the cost of multi-processing are just not justified
    - Output:
        - a dense tensor (stored as multiple sparse/compress tensor) of time-series/cross-section features
    - Hardware:
        - mostly a single CPU thread (with very limited parallelism (you can try, and you will fail :>))
2. **Layer 2**: (for both backtest and trading)
    - Flow:
        - construct(through research/mining) alpha formula/rules from static dense features tensor of layer 1
        - ML(DL/ensemble) layer to construct superalpha from alpha pool
        - DRL(deep-reinforced) layer to do alpha mining
        - standard alpha/CTA research with financial domain knowledge
        - Parameter tuning for strategies
    - Key:
        - each alpha should able to be constructed simply using a few features with a limited amount of operators
    - Good:
        - completely eliminate the need to run Layer 1 for each backtest run, MASSIVELY increase backtest speed
        - because of layer abstraction, add a bit overhead for real trading, but is completely fine for minute-level-trading
        - because all features are ready, fine-grained sync no longer needed, can freely calculate with no worries of:
            - backtest/real-trading discrepancy
            - future informations
    - Output:
        - alpha, super(composite) alpha, trading signal
    - Hardware:
        - GPU clusters (happy now :>)

- For fast backtesting/researching/alpha-mining:
    - for GPU calculation, only dense tensors can support all operations, compared to sparse tensors
    - Size calculation for dense tensor (time-series * cross-section * features):
        - 20years * 24hrs * 1min * 5000assets * 200 features * 16b = 20TiB
        - 20years * 5hrs * 5min * 5000assets * 200 features * 16b = 582GiB
        - 1years * 5hrs * 5min * 5000assets * 100 features * 16b = 14GiB (this is the max batch size for GPU calculation)
    - for tick data, you need to have multiple GPUs and do batch runs daily

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