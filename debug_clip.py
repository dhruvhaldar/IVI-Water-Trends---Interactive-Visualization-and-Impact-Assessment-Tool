
import pandas as pd
import numpy as np

df = pd.DataFrame({'val': [-5]})
print(f"Original: {df['val'].tolist()}")
df['val'] = df['val'].clip(lower=0)
print(f"Clipped: {df['val'].tolist()}")
print(f"Comparison: {df['val'].values >= 0}")
