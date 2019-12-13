# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# -------------------------------Utils
# =============================================================================
def get_data(dir_path, num_of_imgs): 
    X = np.zeros([num_of_imgs, 784])
    filenames = {}
    for filepath in os.listdir(dir_path):
        if filepath.endswith(".jpg"):
            index = int(filepath[:-4]) - 1
            image_array = plt.imread(os.path.join(dir_path, filepath))\
                            .flatten()
            X[index] = image_array
            filenames[index] = filepath.split("/")[-1]
            
#    X = np.insert(X, X.shape[1], values=1, axis=1)
    return X, filenames

def print_progress(iteration_type, iteration_value):
    print( '\r ' + iteration_type + ' %s' % (str(iteration_value)), end = '\r')

    
def draw_plot(x, y, plot_label, img_path=None):
    """

    Parameters
    ----------
    x : 1D numpy array
        x-axis to plot.
    y : 1D numpy array
        y-axis to plot.
    plot_label : string
    img_path : img_path, optional
        if given the image is saved. The default is None.

    Returns
    -------
    None.

    """

    # ax = fig.add_subplot()
    plt.style.use('fivethirtyeight')
    plt.scatter(x, y)
    # ax.set(xlim=(0, num_of_classes),
    #        ylim=(0, num_of_samples_per_class),
    #        option='auto')
    plt.xlabel('Digits')
    plt.ylabel('Max. Count')
    plt.title(plot_label)
    fig = plt.gcf()
    # ax.plot(x, y, color='blue')
    plt.show()
    plt.draw()
    if img_path:
        fig.savefig(img_path, dpi=100, bbox_inches="tight")