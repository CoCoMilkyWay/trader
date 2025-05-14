import pandas as pd
import os

os.environ["MLFINLAB_API_KEY"] = "cmai9u31o0001jj08k3o09s4k"

data = pd.read_csv(
    "https://raw.githubusercontent.com/hudson-and-thames/example-data/main/tick_data.csv"
)
print(data)
