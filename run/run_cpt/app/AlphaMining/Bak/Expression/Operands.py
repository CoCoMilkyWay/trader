import torch
from Mining.Config.Config import cfg
from AlphaMining.Mining.Expression.Content import Value, DimensionType, Dimension

register_operands = [i for i in range(cfg.num_registers)]

# define you constant operands here

scalar_operands = []
for c in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    scalar_operands.append(Value(value=torch.tensor(
        [c], dtype=torch.float16), dimension=Dimension([DimensionType.oscillator])),)

for c in [0.01, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10]:
    scalar_operands.append(Value(value=torch.tensor(
        [c], dtype=torch.float16), dimension=Dimension([DimensionType.ratio])),)

for c in [5, 10, 20, 30, 40, 50, 60, 120, 240]:
    scalar_operands.append(Value(value=torch.tensor(
        [c], dtype=torch.float16), dimension=Dimension([DimensionType.timedelta])),)

vector_operands = {
}

matrix_operands = {
    "open":     Value(value=torch.tensor([]),     dimension=Dimension([DimensionType.price])),
    "close":    Value(value=torch.tensor([]),     dimension=Dimension([DimensionType.price])),
    "high":     Value(value=torch.tensor([]),     dimension=Dimension([DimensionType.price])),
    "low":      Value(value=torch.tensor([]),     dimension=Dimension([DimensionType.price])),
    "vwap":     Value(value=torch.tensor([]),     dimension=Dimension([DimensionType.price])),
    "volume":   Value(value=torch.tensor([]),     dimension=Dimension([DimensionType.volume])),
}
