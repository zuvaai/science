import numpy as np
import pandas as pd
import sys

if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])
    input = data['Input'].mean()
    output = data['Output'].mean()
    print(f"{input:.2f} {output:.2f}")