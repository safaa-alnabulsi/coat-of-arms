#!/usr/bin/python

import os
import pandas as pd
from PIL import Image

if __name__ == "__main__":
    
#     data_location = '/home/space/datasets/COA/generated-data-api-large'
    # data_location =  '/home/space/datasets/COA/generated-data-api-small'
    # data_location =  'baseline-gen-data/large'
    # data_location =  'baseline-gen-data/medium'
    # data_location =  'baseline-gen-data/small'
    # data_location = '/Users/salnabulsi/tub/coat-of-arms/data/cropped_coas/out'
#     data_location = '//home/salnabulsi/coat-of-arms/data/new'
    data_location = '/home/space/datasets/COA/generated-single-simple'

    caption_file = data_location + '/captions_psumsq.txt100x100'
#     caption_file = data_location + '/test_real_captions_psumsq.txt'

    root_folder_images = data_location + '/images/'
    new_image_directory = data_location + '/res_images224x224/'
#     new_image_directory = data_location + '/resized/'
    
    if not os.path.exists(new_image_directory):
        os.mkdir(new_image_directory)
    print("Directory '% s' created" % new_image_directory)
    
    df = pd.read_csv(caption_file)

    for img_name in df['image']:
        image_path =  root_folder_images + img_name
        new_image_path = new_image_directory + img_name
        img = Image.open(image_path)
        
        # resize the image t0 100x100 to improve the iteration time
        crops_size = 224,224
        img.thumbnail(crops_size, Image.ANTIALIAS)
        img.save(new_image_path)
        
        print(f'Image {new_image_path} has been resized!')
