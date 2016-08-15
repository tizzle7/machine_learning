# plot_tools.py

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """Visualize the decision boundaries of a perceptron classifier for
    2D datasets.

    Arguements:
    X -- matrix containing features of several samples
    y -- target values for each sample in an array
    classifier -- fitted Perceptron classifier
    resolution --
    """
    # create figure and axe objects
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    # define markers and colors
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")

    # create colormap
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # determine minimum and maximum values for the two model features
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # create grid arrays for each feature
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # predict the class labels for all grid points, reshape the label vector
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = z.reshape(xx1.shape)

    # draw a contour plot that maps the different decision regions to different colors
    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    ax.set(xlim=(xx1.min(), xx1.max()), ylim=(xx2.min(), xx2.max()))

    # plot the data points
    for i, label in enumerate(np.unique(y)):
        ax.scatter(X[y == label, 0], X[y == label, 1], alpha=0.8,
                    c=cmap(i), marker=markers[i], label=label)

    return fig, ax
    
    
    
    
                           
    

    
    
