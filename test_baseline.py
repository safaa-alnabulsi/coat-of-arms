#!/usr/bin/python

#imports 
import os
import nltk
import spacy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx as onnx
import torch.optim as optim
import torchvision.models as models

from src.baseline.vocabulary import Vocabulary
from src.utils import print_time, list_of_tensors_to_numpy_arr, plot_image, plot_im
from src.accuracy import Accuracy
from src.baseline.coa_model import get_new_model,load_model, train_validate_test_split, init_testing_model, test_model, test_rand_image, get_training_mean_std, get_min_max_acc_images
from pyinstrument import Profiler

import argparse


if __name__ == "__main__":
    print('starting the test script')

    parser = argparse.ArgumentParser(description='A script for training the baseline model')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Full path to the dataset', default='/home/space/datasets/COA/generated-data-api')
    parser.add_argument('--run-name', dest='run_name', type=str, help='Name of the run: e.g. run-{time}')
    parser.add_argument('--model-name', dest='model_name', type=str, help='Name of the trained model: e.g. modelname.pth')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Number of Batch size', default=128)
    parser.add_argument('--local', dest='local', type=str, help='running on local?', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])
    parser.add_argument('--real-data', dest='real_data', type=str, help='testing cropped real dataset?', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])

    args = parser.parse_args()

    data_location = args.dataset
    run_name = args.run_name
    model_name = args.model_name
    batch_size = args.batch_size
    local = args.local
    real_data = args.real_data
            
    if local in ['yes','Yes','y','Y'] :
        local = True
    else: 
        local = False
    
    if local:
        run_path = f"experiments/{run_name}"
    else:
        run_path = f"/home/space/datasets/COA/experiments/{run_name}"
    
    model_path= f"{run_path}/{model_name}"

    if real_data in ['yes','Yes','y','Y'] :
        real_data = True
    else: 
        real_data = False
        
    print('testing cropped real dataset? ',real_data)
    print('running on local ',local)
    print('data_location is ',data_location)
    print('model_path is ',model_path)
    print('batch_size is ',batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # --------------------------------------- test dataset ---------------------------------
    
    print('Dataset exists in', data_location)    
    if real_data:
        root_folder_images = data_location + '/resized'
        test_caption_file  = data_location + '/test_real_captions_psumsq.txt'
    else:
        root_folder_images = data_location + '/images'
        test_caption_file  = data_location + '/test_captions_psumsq.txt'
    
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
    mean, std = get_training_mean_std(run_path)

    # ---------------------------------------- Loaders ------------------------------------------------
    
    test_loader, test_dataset = init_testing_model(test_caption_file, 
                                                   root_folder_images, 
                                                   mean, std,
                                                   NUM_WORKER,
                                                   vocab,
                                                   batch_size, 
                                                   device, 
                                                   pin_memory=False)

    # ----------------------------------------- Hyperparams ---------------------------------

    # Hyperparams

    embed_size=300
    vocab_size = len(test_dataset.vocab)
    attention_dim=256
    encoder_dim=2048
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
    
    # ------------------------------------------ Load model ----------------------------------------
    profiler = Profiler(async_mode='disabled')
    profiler.start()

    model, optimizer, loss,criterion = load_model(model_path, 
                                    hyper_params, 
                                    learning_rate,
                                    drop_prob, 
                                    ignored_idx,
                                    pretrained=True)
    profiler.stop()
    profiler.print()

    # --------------------------------- Testing the model ----------------------------------------
    
    test_losses, accuracy_test_list, image_names_list, acc_test_score, test_loss = test_model(model, 
                                                                            criterion,
                                                                            test_loader, 
                                                                            test_dataset, 
                                                                            vocab_size, 
                                                                            device,
                                                                            run_path,
                                                                            real_data)    
    
    # -------------------------------- Saving the results ----------------------------------------
    
    image_with_max_acc, image_with_min_acc = get_min_max_acc_images(accuracy_test_list, image_names_list)
    print('image name with max accuracy', image_with_max_acc)
    print('image name with max accuracy', image_with_min_acc)

    # torch.cuda.empty_cache()
#     test_rand_image(model, test_dataset, test_loader, device)
    
