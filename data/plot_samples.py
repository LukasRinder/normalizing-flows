import matplotlib.pyplot as plt

def plot_samples_2d(data, name=None):
    plt.figure(figsize=(5,5))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.scatter(data[:, 0], data[:, 1]) #, s=15)
    
    if name:
        plt.savefig(name + ".png", format="png")