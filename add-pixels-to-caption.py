#!/usr/bin/python

#imports 

import os
import itertools
import argparse
from src.caption import Caption
from src.armoria_api_generator_helper import ArmoriaAPIGeneratorHelper


if __name__ == "__main__":
    print('starting the script')
    parser = argparse.ArgumentParser(description='A script for generating armoria dataset')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Full path to the dataset', default='/home/space/datasets/COA/generated-data-api-large')
    parser.add_argument('--index', dest='index', type=int, help='Start index', default=1)

    args = parser.parse_args()
    start_index = args.index
    FOLDER_NAME = args.dataset

    print('Start index', start_index)
    
    caption_file = FOLDER_NAME + '/' + 'captions.txt'   
    new_caption_file = FOLDER_NAME + '/' + 'captions-psumsq.txt'  
    
    api_gen_helper = ArmoriaAPIGeneratorHelper(caption_file, FOLDER_NAME,[],start_index) 

    if start_index == 1:
        api_gen_helper.creat_caption_file(new_caption_file,columns='image,caption,psum,psum_sq')

    api_gen_helper.add_pixels_column(FOLDER_NAME, new_caption_file,caption_file,start_index)