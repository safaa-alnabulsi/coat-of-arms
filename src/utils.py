import matplotlib.image as mpimg
import numpy as np

def plot_bar_chart(data):
    labels, values = zip(*data)
    indexes = np.arange(len(labels))
    width = 0.5
    plt.rcParams["figure.figsize"] = (20, 10)
    plt.bar(indexes, values, width)
    plt.xticks(indexes, labels, rotation='vertical')
    plt.show()


def plot_image(image_file):
    img = mpimg.imread(image_file)
    imgplot = plt.imshow(img)
    plt.show()
