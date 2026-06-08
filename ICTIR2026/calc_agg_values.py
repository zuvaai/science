import numpy as np
import pandas as pd
import sys

if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])
    mean = data['Value'].mean()
    std = data['Value'].std()
    print(f"{mean:.3f} ({std:.3f})")