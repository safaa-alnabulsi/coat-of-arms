#!/usr/bin/python

#imports 

import os
import itertools
import argparse
from src.caption import Caption
from src.armoria_api_generator_helper import ArmoriaAPIGeneratorHelper

def resize_split_file(old_caption_file, new_caption_file, folder_name, start_index, images_folder):

    api_gen_helper = ArmoriaAPIGeneratorHelper(old_caption_file, folder_name,[],start_index) 

    if start_index == 0:
        api_gen_helper.creat_caption_file(new_caption_file,columns='image,caption,psum,psum_sq')

    api_gen_helper.add_pixels_column(folder_name, new_caption_file, old_caption_file,start_index, images_folder)

if __name__ == "__main__":
    print('starting the script')
    parser = argparse.ArgumentParser(description='A script for generating armoria dataset')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Full path to the dataset', default='/home/space/datasets/COA/generated-single-simple')
    parser.add_argument('--index', dest='index', type=int, help='Start index', default=0)

    args = parser.parse_args()
    start_index = args.index
    FOLDER_NAME = args.dataset

    print('Start index', start_index)
    
#     caption_file = FOLDER_NAME + '/' + 'test_real_captions.txt'   
#     new_caption_file = FOLDER_NAME + '/' + 'test_real_captions_psumsq.txt'  

    old_caption_file = FOLDER_NAME + '/' + 'captions_psumsq.txt100x100'   
#     new_caption_file = FOLDER_NAME + '/' + 'captions_psumsq.txt100x100'      
#     images_folder = 'res_images100x100'

    new_caption_file = FOLDER_NAME + '/' + 'captions_psumsq.txt224x224'      
    images_folder = 'res_images224x224'

    resize_split_file(old_caption_file, new_caption_file, FOLDER_NAME, start_index, images_folder)
    
    
    # we need to keep the same split above so we can compare accuracy
    
    # fix this as it is our base
    old_train_annotation_file = FOLDER_NAME + '/train_captions_psumsq.txt100x100'
    old_val_annotation_file  = FOLDER_NAME + '/val_captions_psumsq.txt100x100'
    old_test_annotation_file  = FOLDER_NAME + '/test_captions_psumsq.txt100x100'

#     # original 500x500
#     new_train_annotation_file = FOLDER_NAME + '/train_captions_psumsq.txt'
#     new_val_annotation_file  = FOLDER_NAME + '/val_captions_psumsq.txt'
#     new_test_annotation_file  = FOLDER_NAME + '/test_captions_psumsq.txt'
#     images_folder = 'images'
    
#     resize_split_file(old_train_annotation_file, new_train_annotation_file, FOLDER_NAME, start_index, images_folder)
#     resize_split_file(old_val_annotation_file, new_val_annotation_file, FOLDER_NAME, start_index, images_folder)
#     resize_split_file(old_test_annotation_file, new_test_annotation_file, FOLDER_NAME, start_index, images_folder)


    # resnet size 224x224
    new_train_annotation_file = FOLDER_NAME + '/train_captions_psumsq.txt224x224'
    new_val_annotation_file  = FOLDER_NAME + '/val_captions_psumsq.txt224x224'
    new_test_annotation_file  = FOLDER_NAME + '/test_captions_psumsq.txt224x224'
    images_folder = 'res_images224x224'
    
    resize_split_file(old_train_annotation_file, new_train_annotation_file, FOLDER_NAME, start_index, images_folder)
    resize_split_file(old_val_annotation_file, new_val_annotation_file, FOLDER_NAME, start_index, images_folder)
    resize_split_file(old_test_annotation_file, new_test_annotation_file, FOLDER_NAME, start_index, images_folder)


#     # 200x200
#     new_train_annotation_file = FOLDER_NAME + '/train_captions_psumsq.txt200x200'
#     new_val_annotation_file  = FOLDER_NAME + '/val_captions_psumsq.txt200x200'
#     new_test_annotation_file  = FOLDER_NAME + '/test_captions_psumsq.txt200x200'
#     images_folder = 'res_images200x200'
    
#     resize_split_file(old_train_annotation_file, new_train_annotation_file, FOLDER_NAME, start_index, images_folder)
#     resize_split_file(old_val_annotation_file, new_val_annotation_file, FOLDER_NAME, start_index, images_folder)
#     resize_split_file(old_test_annotation_file, new_test_annotation_file, FOLDER_NAME, start_index, images_folder)
    
    
#     # 300x300
#     new_train_annotation_file = FOLDER_NAME + '/train_captions_psumsq.txt300x300'
#     new_val_annotation_file  = FOLDER_NAME + '/val_captions_psumsq.txt300x300'
#     new_test_annotation_file  = FOLDER_NAME + '/test_captions_psumsq.txt300x300'
#     images_folder = 'res_images300x300'
    
#     resize_split_file(old_train_annotation_file, new_train_annotation_file, FOLDER_NAME, start_index, images_folder)
#     resize_split_file(old_val_annotation_file, new_val_annotation_file, FOLDER_NAME, start_index, images_folder)
#     resize_split_file(old_test_annotation_file, new_test_annotation_file, FOLDER_NAME, start_index, images_folder)
    
    
#     # 400x400
#     new_train_annotation_file = FOLDER_NAME + '/train_captions_psumsq.txt400x400'
#     new_val_annotation_file  = FOLDER_NAME + '/val_captions_psumsq.txt400x400'
#     new_test_annotation_file  = FOLDER_NAME + '/test_captions_psumsq.txt400x400'
#     images_folder = 'res_images400x400'
    
#     resize_split_file(old_train_annotation_file, new_train_annotation_file, FOLDER_NAME, start_index, images_folder)
#     resize_split_file(old_val_annotation_file, new_val_annotation_file, FOLDER_NAME, start_index, images_folder)
#     resize_split_file(old_test_annotation_file, new_test_annotation_file, FOLDER_NAME, start_index, images_folder)
    