from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

def plot_samples_2d(data: np.ndarray, name: Optional[str] = None) -> None:
    plt.figure(figsize=(5,5))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.scatter(data[:, 0], data[:, 1]) #, s=15)
    
    if name:
        plt.savefig(name + ".png", format="png")