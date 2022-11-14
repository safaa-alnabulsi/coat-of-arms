#!/usr/bin/python

#imports 
import os
import nltk
import spacy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx as onnx
import torch.optim as optim
import torchvision.models as models

# imports 
import torch
import pandas as pd

from src.baseline.vocabulary import Vocabulary
from src.baseline.data_loader import get_loader, get_mean, get_std
from src.baseline.coa_model import get_new_model, init_testing_model,test_model,get_min_max_acc_images
from src.accuracy import WEIGHT_MAP_ONLY_CHARGE

import argparse


if __name__ == "__main__":
    print('starting the test vanilla script')

    parser = argparse.ArgumentParser(description='A script for training the baseline model')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Full path to the dataset', default='/home/space/datasets/COA/generated-data-api')
    parser.add_argument('--real-data', dest='real_data', type=str, help='testing cropped real dataset?', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])
    parser.add_argument('--pretrained', dest='pretrained', type=str, help='vanilla pretrained?', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])

    args = parser.parse_args()
    
    seed = 10
    # random.seed(seed)     # python random generator
    # np.random.seed(seed)  # numpy random generator
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
    data_location = args.dataset
    real_data = args.real_data
    pretrained = args.pretrained

    if pretrained in ['yes','Yes','y','Y'] :
        pretrained = True
    else:
        pretrained = False
            
    if real_data in ['yes','Yes','y','Y'] :
        real_data = True
        img_h=620
        img_w=600
    else: 
        real_data = False
        img_h=500
        img_w=500
        
    print('testing cropped real dataset? ',real_data)
    print('data_location is ',data_location)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # --------------------------------------- test dataset ---------------------------------
    
    print('Dataset exists in', data_location)    
    if real_data:
        root_folder_images = data_location + '/resized'
        test_caption_file  = data_location + '/test_real_captions_psumsq.txt'
    else:
        root_folder_images = data_location + '/images'
        # test_caption_file  = data_location + '/test_captions_psumsq.txt'
        test_caption_file  = data_location + '/real_captions_psumsq_lions_cleaned.txt'

    df = pd.read_csv(test_caption_file)
    print("There are {} test images".format(len(df)))

    # ----------------------------------------- Constants -----------------------------------------
    
    #setting the constants
    NUM_WORKER = 2 #### this needs multi-core
    freq_threshold = 5
    
    # 30 minutes to create those, as it's baseline, i ran it several times and it's the same
    vocab = Vocabulary(freq_threshold)
    vocab.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'lion': 4, 'rampant': 5, 'passt': 6, 'guard': 7, 'head': 8, 'lions': 9, 'cross': 10, 'molxine': 11, 'patonce': 12, 'eagle': 13, 'doubleheaded': 14, 'eagles': 15, 'a': 16, 'b': 17, 'o': 18, 's': 19, 'g': 20, 'e': 21, 'v': 22, '1': 23, '2': 24, '3': 25, '4': 26, '5': 27, '6': 28, '7': 29, '8': 30, '9': 31, '10': 32, '11': 33, 'border': 34, '&': 35}
    vocab.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>', 4: 'lion', 5: 'rampant', 6: 'passt', 7: 'guard', 8: 'head', 9: 'lions', 10: 'cross', 11: 'moline', 12: 'patonce', 13: 'eagle', 14: 'doubleheaded', 15: 'eagles', 16: 'a', 17: 'b', 18: 'o', 19: 's', 20: 'g', 21: 'e', 22: 'v', 23: '1', 24: '2', 25: '3', 26: '4', 27: '5', 28: '6', 29: '7', 30: '8', 31: '9', 32: '10', 33: '11', 34: 'border', 35: '&'}
    
    # ------------------------------------------ Get mean & std ---------------------------------------
    mean=0.3272075951099396
    std=0.3805903494358063
    batch_size = 256

   # ---------------------------------------- Loaders ------------------------------------------------
    
    test_loader, test_dataset = init_testing_model(test_caption_file, 
                                                   root_folder_images, 
                                                   mean, std,
                                                   NUM_WORKER,
                                                   vocab,
                                                   batch_size, 
                                                   device, 
                                                   pin_memory=False,
                                                   img_h=img_h, 
                                                   img_w=img_w)

    # ----------------------------------------- Hyperparams ---------------------------------
    # Hyperparams

    embed_size=300
    vocab_size = len(test_dataset.vocab)
    attention_dim=256
    # encoder_dim=2048  ### resnet50
    encoder_dim=512  ### resnet34 & resnet18
    decoder_dim=512
    learning_rate = 3e-4
    drop_prob=0.3
    ignored_idx = test_dataset.vocab.stoi["<PAD>"]

    hyper_params = {'embed_size': embed_size,
                    'attention_dim': attention_dim,
                    'encoder_dim': encoder_dim,
                    'decoder_dim': decoder_dim,
                    'vocab_size': vocab_size
                  }
    
    print('hyper_params: ',hyper_params)
    
    # ------------------------------------------ Get new model ----------------------------------------
    pretrained = pretrained
    model, optimizer, criterion = get_new_model(hyper_params, 
                                                learning_rate, 
                                                ignored_idx, drop_prob, device, 
                                                pretrained)
    


    # --------------------------------- Testing the model ----------------------------------------
    test_losses, accuracy_test_list, image_names_list, predictions_list, acc_test_score, test_loss = test_model(model, 
                                                                            criterion,
                                                                            test_loader, 
                                                                            test_dataset, 
                                                                            vocab_size, 
                                                                            device,
                                                                            'vanilla-logs',
                                                                            real_data,
                                                                            weights_map=WEIGHT_MAP_ONLY_CHARGE)    


    print("test_loss: ", test_loss)
    print("acc_test_score: ", acc_test_score)
    
    # -------------------------------- Saving the results ----------------------------------------
    
    image_with_min_acc, image_with_max_acc = get_min_max_acc_images(accuracy_test_list, image_names_list, predictions_list)
    print('images with highest accuracy', image_with_max_acc)
    print('images with lowest accuracy', image_with_min_acc)


    # torch.cuda.empty_cache()
#     test_rand_image(model, test_dataset, test_loader, device)
    