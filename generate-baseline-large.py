#!/usr/bin/python

#imports 

import os
import itertools
import argparse

import src.alphabet as alphabet
from src.caption import Caption
from src.armoria_api_generator_helper import ArmoriaAPIGeneratorHelper
from src.armoria_api import ArmoriaAPIPayload, ArmoriaAPIWrapper, COLORS_MAP, SINGLE_LION_MODIFIERS_MAP, PLURAL_LION_MODIFIERS_MAP, CROSS_MODIFIERS_MAP, \
SINGLE_EAGLE_MODIFIERS_MAP, PLURAL_EAGLE_MODIFIERS_MAP, \
POSITIONS, SIZES, NUMBERS, NUMBERS_MULTI, SINGLE_POSITION


if __name__ == "__main__":
    print('starting the script')
    BORDER_MOD = ['& border', '']
    parser = argparse.ArgumentParser(description='A script for generating armoria dataset')
    parser.add_argument('--index', dest=index, type=int, help='Start index', default=1)

    args = parser.parse_args()
    start_index = args.index
    print('Start index', start_index)

    ## Single object
    # lion, modifiers and colors
    permutations1 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list( SINGLE_LION_MODIFIERS_MAP.keys()),BORDER_MOD)))

    # cross, modifiers and colors
    permutations2 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list( CROSS_MODIFIERS_MAP.keys()),BORDER_MOD)))

    # eagle, modifiers and colors
    permutations3 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list( SINGLE_EAGLE_MODIFIERS_MAP.keys()),BORDER_MOD)))

    possible_single_permutations = permutations1 + permutations2 + permutations3

    print('Total number of permutations:', len(possible_single_permutations))

    ## Plural Object with Number

    # lion, modifiers and colors
    permutations1 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list(NUMBERS), list( PLURAL_LION_MODIFIERS_MAP.keys()),BORDER_MOD)))

    # eagle, modifiers and colors
    permutations2 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list(NUMBERS), list( PLURAL_EAGLE_MODIFIERS_MAP.keys()),BORDER_MOD)))

    possible_pl_permutations = permutations1 + permutations2

    print('Total number of plural permutations:', len(possible_pl_permutations))

    ## Multi Objects - single


    # lion & eagle
    permutations1 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()),list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list( SINGLE_LION_MODIFIERS_MAP.keys()), list( SINGLE_EAGLE_MODIFIERS_MAP.keys()),BORDER_MOD)))

    # lion & cross
    permutations2 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()),list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list( SINGLE_LION_MODIFIERS_MAP.keys()), list( CROSS_MODIFIERS_MAP.keys()),BORDER_MOD)))

    # eagle & cross
    permutations3 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()),list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list( SINGLE_EAGLE_MODIFIERS_MAP.keys()), list( CROSS_MODIFIERS_MAP.keys()),BORDER_MOD)))

    # lion & cross & eagle
    permutations4 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()),list(COLORS_MAP.keys()),list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list( SINGLE_LION_MODIFIERS_MAP.keys()), list( SINGLE_EAGLE_MODIFIERS_MAP.keys()), list( CROSS_MODIFIERS_MAP.keys()),BORDER_MOD)))

    possible_multi_single_permutations = permutations1 + permutations2 + permutations3 + permutations4

    print('Total number of permutations:', len(possible_multi_single_permutations))


    ## Multi Objects - plural

    # lion & eagle
    permutations1 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()),list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list(NUMBERS_MULTI),list( PLURAL_LION_MODIFIERS_MAP.keys()), list(NUMBERS_MULTI), list( PLURAL_EAGLE_MODIFIERS_MAP.keys()),BORDER_MOD)))

    # lion & cross
    permutations2 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()),list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list(NUMBERS_MULTI),list( PLURAL_LION_MODIFIERS_MAP.keys()), list( CROSS_MODIFIERS_MAP.keys()),BORDER_MOD)))

    # eagle & cross
    permutations3 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()),list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list(NUMBERS_MULTI),list( PLURAL_EAGLE_MODIFIERS_MAP.keys()), list( CROSS_MODIFIERS_MAP.keys()),BORDER_MOD)))

    # lion & cross & eagle
    permutations4 = list(dict.fromkeys(itertools.product(list(COLORS_MAP.keys()),list(COLORS_MAP.keys()),list(COLORS_MAP.keys()), list(COLORS_MAP.keys()), list(NUMBERS_MULTI), list( PLURAL_LION_MODIFIERS_MAP.keys()), list(NUMBERS_MULTI),list( PLURAL_EAGLE_MODIFIERS_MAP.keys()), list( CROSS_MODIFIERS_MAP.keys()),BORDER_MOD)))

    possible_multi_plural_permutations = permutations1 + permutations2 + permutations3 + permutations4

    print('Total number of plural permutations:', len(possible_multi_plural_permutations))

    total_possible_permutations = possible_single_permutations + possible_pl_permutations + \
                                possible_multi_single_permutations  + possible_multi_plural_permutations

    FOLDER_NAME = '/home/space/datasets/COA/generated-data-api-large'
    # FOLDER_NAME = '../generated'
    caption_file = FOLDER_NAME + '/' + 'captions.txt'
    api_gen_helper = ArmoriaAPIGeneratorHelper(caption_file, FOLDER_NAME, total_possible_permutations, start_index)   
    
    api_gen_helper.generate_dataset()
