from typing import Dict, List
import matplotlib.pyplot as plt

def plot(out_file: str, curve_names: List, curves: List, labels: Dict):
    for curve_name, curve in zip(curve_names, curves):
        length = len(curve)
        x_axis = range(1, length+1)
        plt.plot(x_axis, curve, label=curve_name)
        plt.ylabel(labels["y"])
        plt.xlabel(labels["x"])
        plt.legend(loc='upper right')
        
    plt.savefig(out_file)

if __name__ == "__main__":
    plot()