#!/usr/bin/python

#imports 
import os
from xmlrpc.client import boolean
import torch
import spacy
import pandas as pd
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import nltk
import numpy as np
import torch.nn as nn
import torch.onnx as onnx
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from time import sleep
from src.baseline.vocabulary import Vocabulary
from src.baseline.data_loader import get_loader, get_loaders, get_mean, get_std
from src.accuracy import Accuracy
from src.baseline.coa_model import save_model, get_new_model, train_model, train_validate_test_split
import torch.multiprocessing as mp
from src.utils import print_time

from pyinstrument import Profiler
from datetime import datetime

import argparse


if __name__ == "__main__":
    print('starting the script')

    parser = argparse.ArgumentParser(description='A script for training the baseline model')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Full path to the dataset', default='/home/space/datasets/COA/generated-data-api')
    parser.add_argument('--epochs', dest='epochs', type=int, help='Number of epochs',default=10)
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Number of Batch size', default=128)
    parser.add_argument('--resplit', dest='resplit', type=str, help='resplit the samples', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])
    parser.add_argument('--local', dest='local', type=str, help='running on local?', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])

    args = parser.parse_args()

    data_location = args.dataset
    batch_size = args.batch_size
    num_epochs = args.epochs
    resplit = args.resplit 
    local = args.local
    
    if resplit in ['yes','Yes','y','Y'] :
        resplit = True
    else: 
        resplit = False
        
    if local in ['yes','Yes','y','Y'] :
        local = True
    else: 
        local = False

    # get the timestamp to create default logsdir
    now = datetime.now() # current date and time
    timestr = now.strftime("%m-%d-%Y-%H:%M:%S")
    if local:
        model_folder = f"experiments/run-{timestr}"
    else:
        model_folder = f"/home/space/datasets/COA/experiments/run-{timestr}"
                
    # create the folder where all models files will be stored
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    print('running on local ',local)
    print('data_location is ',data_location)
    print('model_folder is ',model_folder)
    print('batch_size is ',batch_size)
    print('num_epochs is ',num_epochs)
    print('resplit is ',resplit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    print('Dataset exists in', data_location)    
    caption_file = data_location + '/captions-psumsq.txt'
    root_folder_images = data_location + '/images'
    df = pd.read_csv(caption_file)

    train_annotation_file = data_location + '/train_captions_psumsq.txt'
    val_annotation_file  = data_location + '/val_captions_psumsq.txt'
    test_annotation_file  = data_location + '/test_captions_psumsq.txt'
    
    if resplit:
        train, validate, test = train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None)
        train.to_csv(train_annotation_file, sep=',',index=False)
        test.to_csv(test_annotation_file, sep=',',index=False)
        validate.to_csv(val_annotation_file, sep=',',index=False)

    print("There are {} total images".format(len(df)))

    df1 = pd.read_csv(train_annotation_file)
    print("There are {} train images".format(len(df1)))

    df2 = pd.read_csv(val_annotation_file)
    print("There are {} val images".format(len(df2)))

    df3 = pd.read_csv(test_annotation_file)
    print("There are {} test images".format(len(df3)))
    
    # -------------------------------------------------------------------------------------------------------
    
    #setting the constants
    NUM_WORKER = 2 #### this needs multi-core
    freq_threshold = 5
    
    # 30 minutes to create those, as it's baseline, i ran it several times and it's the same
    vocab = Vocabulary(freq_threshold)
    vocab.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'lion': 4, 'rampant': 5, 'passt': 6, 'guard': 7, 'head': 8, 'lions': 9, 'cross': 10, 'moline': 11, 'patonce': 12, 'eagle': 13, 'doubleheaded': 14, 'eagles': 15, 'a': 16, 'b': 17, 'o': 18, 's': 19, 'g': 20, 'e': 21, 'v': 22, '1': 23, '2': 24, '3': 25, '4': 26, '5': 27, '6': 28, '7': 29, '8': 30, '9': 31, '10': 32, '11': 33, 'border': 34, '&': 35}
    vocab.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>', 4: 'lion', 5: 'rampant', 6: 'passt', 7: 'guard', 8: 'head', 9: 'lions', 10: 'cross', 11: 'moline', 12: 'patonce', 13: 'eagle', 14: 'doubleheaded', 15: 'eagles', 16: 'a', 17: 'b', 18: 'o', 19: 's', 20: 'g', 21: 'e', 22: 'v', 23: '1', 24: '2', 25: '3', 26: '4', 27: '5', 28: '6', 29: '7', 30: '8', 31: '9', 32: '10', 33: '11', 34: 'border', 35: '&'}
    
    # ------------------------------------------ initial train loader --------------------------------------------------------
    print_time('\n ------------------------ \n calling get_loader with calc_mean=True - for mean')
    profiler = Profiler(async_mode='disabled')
    profiler.start()

    train_loader, train_dataset = get_loader(
        root_folder=root_folder_images,
        annotation_file=train_annotation_file,
        transform=None,  # <=======================
        num_workers=NUM_WORKER,
        vocab=vocab,
        batch_size=batch_size,
        pin_memory=False,
        calc_mean=True
    )
    
    profiler.stop()
    profiler.print()

    # ------------------------------------------ Calc mean -------------------------------------------------------------
    profiler = Profiler(async_mode='disabled')
    profiler.start()

    print_time('\n ------------------------ \n calling get_mean')

    mean = get_mean(train_dataset, train_loader, 500 , 500)

    mean_file = f'{model_folder}/mean.txt'
    with open(mean_file, 'w') as file:
        file.write(str(float(mean)))

    print_time(f'finished calculating the mean: {mean} and saved it to file: {mean_file}')

    profiler.stop()
    profiler.print()

    # ----------------------------------------- Calc std --------------------------------------------------
    profiler = Profiler(async_mode='disabled')
    profiler.start()

    print_time('\n ------------------------ \n calling get_std')

    std = get_std(train_dataset, train_loader, mean)

    std_file = f'{model_folder}/std.txt'
    with open(std_file, 'w') as file:
        file.write(str(float(std)))

    print_time(f'finished calculating the std: {std} and saved it to file: {std_file}')

    profiler.stop()
    profiler.print()

    # ---------------------------------------- Loaders ---------------------------------------------------------------
    torch.cuda.empty_cache()

    # Defining the transform to be applied
#     mp.set_start_method('spawn')

    transform = T.Compose([
        T.Resize(226),                     
        T.RandomCrop(224),                 
        T.ToTensor(),                               
        T.Normalize(mean, std) 
    ])

    print_time('writing the dataloader')
    
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_loaders(
        root_folder=root_folder_images,
        train_annotation_file=train_annotation_file,
        val_annotation_file=val_annotation_file,
        test_annotation_file=test_annotation_file,
        transform=transform,
        num_workers=NUM_WORKER,
        vocab=vocab,
        batch_size=batch_size,
        device=device,
        pin_memory=False
    )

    print_time('finished writing the dataloader')
    # ----------------------------------------- Hyperparams --------------------------------------------------------------

    # Hyperparams
    embed_size=300
    vocab_size = len(train_dataset.vocab)
    attention_dim=256
    encoder_dim=2048
    decoder_dim=512
    learning_rate = 0.01 # 3e-4
    drop_prob=0.3
    ignored_idx = train_dataset.vocab.stoi["<PAD>"]

    hyper_params = {'embed_size': embed_size,
                    'attention_dim': attention_dim,
                    'encoder_dim': encoder_dim,
                    'decoder_dim': decoder_dim,
                    'vocab_size': vocab_size
                  }
    
    print('hyper_params: ',hyper_params)
    # --------------------------------------- Training the model ------------------------------------------------------
    
    # Training the model
    
    print('initialize new model, loss etc')    
    model, optimizer, criterion = get_new_model(hyper_params, learning_rate, ignored_idx, drop_prob, device, True)

    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 20

    model, train_loss, valid_loss, avg_acc, bleu_score = train_model(model, optimizer, criterion, train_dataset, train_loader, val_loader, val_dataset, vocab_size, batch_size, patience, num_epochs, device, model_folder)

    final_accuracy = np.average(avg_acc)
    final_train_loss = np.average(train_loss)
    
    print('Bleu Score: ', bleu_score/8091)
    print('Final accuracy: ', final_accuracy)
    
    # ---------------------------------------- Saving the model ----------------------------------------------------

    # save the latest model
    now = datetime.now() # current date and time
    timestr = now.strftime("%m-%d-%Y-%H:%M:%S")
    model_full_path = f"{model_folder}/baseline-model-{timestr}.pth"

    save_model(model, optimizer, final_train_loss, final_accuracy, model_full_path, hyper_params)
    print('The trained model has been saved to ', model_full_path)
