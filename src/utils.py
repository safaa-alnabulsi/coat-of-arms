import os
import zipfile
import matplotlib.image as mpimg
import numpy as np
from datetime import datetime

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


def print_time(text):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("{} @ Time = {}".format(text, current_time))
    
       
def zip_dir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))


def zip_dir_to_file(path_to_zipped_file, data_folder)
    zipf = zipfile.ZipFile(path_to_zipped_file, 'w', zipfile.ZIP_DEFLATED)
    zip_dir(data_folder, zipf)
    zipf.close()

    
def unzip_file_to_dir(path_to_zipped_file, unzip_location):
    with zipfile.ZipFile(path_to_zipped_file, 'r') as zip_ref:
        zip_ref.extractall(unzip_location)
