import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook as cbook

def visual_AL(LLAL, Rand, title, xlabel, ylabel):
    mu1 = LLAL.mean(axis = 0)
    mu2 = Rand.mean(axis = 0)
    sigma1 = LLAL.std(axis = 0)
    sigma2 = Rand.std(axis = 0)
    # plot it!
    fig, ax = plt.subplots(1)
    ax.plot(t, mu1, label='LLAL', color='red', marker='o')
    ax.plot(t, mu2, label='Rand', color='black', marker='o')
    ax.fill_between(t, mu1+sigma1, mu1-sigma1, facecolor='red', alpha = 0.25)
    ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='black', alpha = 0.25)
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

Nsteps = 10
t = ["1k", "2k", "3k", "4k", "5k", "6k", "7k", "8k", "9k", "10k"]

LLAL = np.array([[47.10, 64.85, 75.07, 74.37, 81.16, 84.76, 86.19, 87.57, 88.92, 89.60],
        [45.87, 59.15, 68.86, 77.17, 80.66, 84.65, 86.74, 88.65, 89.22, 90.00],
        [45.99, 64.29, 69.78, 80.93, 82.06, 85.93, 86.84, 88.84, 89.63, 89.51],
        [49.93, 60.94, 70.35, 75.49, 82.62, 83.29, 86.79, 87.52, 88.52, 89.76],
        [48.71, 62.15, 67.96, 75.42, 81.16, 82.67, 86.72, 87.61, 89.65, 89.20]])
Rand = np.array([[49.05, 55.29, 68.04, 72.24, 78.51, 82.07, 84.91, 84.89, 87.50, 87.89],
        [45.66, 57.56, 65.37, 76.09, 79.56, 82.57, 83.73, 85.92, 86.38, 87.82],
        [47.70, 60.24, 69.01, 75.51, 79.03, 82.32, 84.26, 84.70, 85.94, 86.76],
        [50.25, 63.31, 65.12, 76.25, 79.03, 82.88, 84.39, 85.83, 87.38, 87.29],
        [44.93, 60.36, 71.14, 76.68, 80.44, 82.23, 83.88, 85.63, 86.61, 87.92]])

visual_AL(LLAL, Rand, "Active learning results of image classification over CIFAR-10.", 'Number of labeled images', 'Accuracy (mean of 5 trials)')

plt.show()

LLAL_map = np.array([[0.512, 0.601, 0.644, 0.668, 0.696, 0.709, 0.716, 0.729, 0.736, 0.744],
                     [0.477, 0.591, 0.646, 0.669, 0.688, 0.712, 0.713, 0.725, 0.733, 0.744]])
Rand_map = np.array([[0.497, 0.591, 0.640, 0.654, 0.675, 0.682, 0.693, 0.708, 0.718, 0.722],
                     [0.490, 0.578, 0.629, 0.657, 0.677, 0.690, 0.703, 0.713, 0.714, 0.722]])

visual_AL(LLAL_map, Rand_map, "Active learning results of object detection over PASCAL VOC 2007+2012.", 'Number of labeled images', 'Mean Average Precision (mAP) (mean of 3 trials)')

plt.show()