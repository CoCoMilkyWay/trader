import os
import sys
import subprocess
import numpy as np
import pandas as pd
from typing import List

from dtype import time_bar_dtype, run_bar_dtype
from plot import *

dir = os.path.dirname(__file__)

try:
    subprocess.run(["C:/msys64/usr/bin/bash", "build.sh"], cwd=os.path.join(os.path.dirname(__file__), "cpp"), check=True)
except:
    sys.exit(1)


def main():
    data_path = os.path.join(dir, "data/bars.parquet")

    input_dtype = np.dtype([(k, v) for k, v in time_bar_dtype.items()])
    output_dtype = np.dtype([(k, v) for k, v in run_bar_dtype.items()])

    time_bar = pd.read_parquet(data_path).reset_index(drop=True)
    input_array = time_bar.to_records(index=False).astype(input_dtype)
    input_bytes = input_array.tobytes()

    print(time_bar)

    from cpp import Pipeline  # type: ignore
    try:
        output_bytes = Pipeline.process_bars(input_bytes, input_array.shape[0])
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Caught exception:", str(e))

    vrun_bar = pd.DataFrame(np.frombuffer(output_bytes, dtype=output_dtype))
    print(vrun_bar)
    vrun_bar['return'] = np.log1p(vrun_bar['close'].pct_change().fillna(0))
    
    # plot_3D(stats, [0, 1, 2], idx_weighted)

    limit = int(60/5*24*5*4*1)
    time_bar = time_bar[-limit*5:]
    vrun_bar = vrun_bar[-limit:]

    # plot_line(vrun_bar, 'time', 'close', 'label_discrete')
    # plot_density(vrun_bar, 'label_continuous')
    # plot_density(vrun_bar, 'return')
    # plot_3D(vrun_bar, ['umap_x', 'umap_y', 'umap_z'], 'label_discrete')


if __name__ == "__main__":
    main()
