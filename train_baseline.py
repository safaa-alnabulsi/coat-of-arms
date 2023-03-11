#!/usr/bin/python

#imports 
import os
from xmlrpc.client import boolean
import torch
import random
import pandas as pd
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.onnx as onnx
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from time import sleep
from src.baseline.noise import Noise
from src.baseline.vocabulary import Vocabulary
from src.baseline.data_loader import get_loader, get_loaders, get_mean, get_std
from src.accuracy import WEIGHT_MAP, WEIGHT_MAP_ONLY_SHIELD_COLOR, WEIGHT_MAP_ONLY_CHARGE, WEIGHT_MAP_ONLY_CHARGE_COLOR
from src.baseline.coa_model import save_model, get_new_model, train_model, train_validate_test_split, load_model_checkpoint, load_model
import torch.multiprocessing as mp
from src.utils import print_time

from pyinstrument import Profiler
from datetime import datetime

import argparse


if __name__ == "__main__":
    print('starting the training script')

    parser = argparse.ArgumentParser(description='A script for training the baseline model')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Full path to the dataset', default='/home/space/datasets/COA/generated-data-api')
    parser.add_argument('--epochs', dest='epochs', type=int, help='Number of epochs',default=10)
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Number of Batch size', default=128)
    parser.add_argument('--resplit', dest='resplit', type=str, help='resplit the samples', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])
    parser.add_argument('--local', dest='local', type=str, help='running on local?', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])
    parser.add_argument('--resized-images', dest='resized_images', type=str, help='smaller resized images?', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, help='continue training from last saved checkpoint? yes, will load the model and no will create a new empty model', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])
    parser.add_argument('--run-folder', dest='run_folder', type=str, help='Run Folder name where checkpoint.pt file exists', default='')
    parser.add_argument('--accuracy', dest='accuracy', type=str, help='type of accuracy', default='all', choices=['all','charge-mod-only','charge-color-only','shield-color-only'])
    parser.add_argument('--seed', dest='seed', type=int, help='reproducibility seed', default=1234)
    parser.add_argument('--caption-file', dest='caption_file', type=str, help='caption file for train images', default='captions-psumsq.txt')
    parser.add_argument('--real-data', dest='real_data', type=str, help='training on cropped real dataset?', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])
    parser.add_argument('--baseline-model', dest='baseline_model', type=str, help='baseline_model file name', default='')

    args = parser.parse_args()

    # ---------------------- Reproducibility -------------------
    
    seed = args.seed
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator

    torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # ----------------------------------------------------------------- 

    data_location = args.dataset
    batch_size = args.batch_size
    num_epochs = args.epochs
    resplit = args.resplit 
    local = args.local
    resized_images = args.resized_images
    continue_from_checkpoint = args.checkpoint
    run_folder = args.run_folder
    accuracy = args.accuracy
    caption_file = args.caption_file
    real_data = args.real_data
    baseline_model = args.baseline_model
    
    if resplit in ['yes','Yes','y','Y'] :
        resplit = True
    else: 
        resplit = False
        
    if local in ['yes','Yes','y','Y'] :
        local = True
    else: 
        local = False

    if resized_images in ['yes','Yes','y','Y'] :
        resized_images = True
    else: 
        resized_images = False
        
    if real_data in ['yes','Yes','y','Y'] :
        real_data = True
    else: 
        real_data = False

    if continue_from_checkpoint in ['yes','Yes','y','Y'] :
        continue_from_checkpoint = True
    else: 
        continue_from_checkpoint = False

    # get the timestamp to create default logsdir
    now = datetime.now() # current date and time
    timestr = now.strftime("%m-%d-%Y-%H:%M:%S")
   
    if local:
        path_to_model =  "experiments"
    else:
        path_to_model = "/home/space/datasets/COA/experiments"
        
    if continue_from_checkpoint: 
        model_folder = f"{path_to_model}/{run_folder}"
    else:
        model_folder = f"{path_to_model}/run-{timestr}"
  
    # choices=['all','charge-mod-only','charge-color-only','shield-color-only']  
    weights_map = WEIGHT_MAP           
    if accuracy == 'charge-mod-only':
        weights_map = WEIGHT_MAP_ONLY_CHARGE
    elif accuracy == 'charge-color-only':
        weights_map = WEIGHT_MAP_ONLY_CHARGE_COLOR
    elif accuracy == 'shield-color-only':
        weights_map = WEIGHT_MAP_ONLY_SHIELD_COLOR        
        
    # create the folder where all models files will be stored
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    print('running on local ',local)
    print('data_location is ',data_location)
    print('model_folder is ',model_folder)
    print('training on cropped real dataset? ',real_data)
    print('batch_size is ',batch_size)
    print('num_epochs is ',num_epochs)
    print('resplit is ',resplit)
    print(f'accuracy is {accuracy} and the weights_map is {weights_map}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device is ',device)

    # ---------------------------------------------------------------------

    print('Dataset exists in', data_location)    
    
    train_caption_file  = data_location + '/' + caption_file
    print('Training caption file path: ', train_caption_file)    

    if real_data:
        root_folder_images = data_location + '/resized'
    else:
        if resized_images:
            root_folder_images = data_location + '/res_images'            
        else:
            root_folder_images = data_location + '/images'

    df = pd.read_csv(train_caption_file)

    train_annotation_file = data_location + '/train_captions_psumsq.txt'
    val_annotation_file  = data_location + '/val_captions_psumsq.txt'
    test_annotation_file  = data_location + '/test_captions_psumsq.txt'

    
#     train_annotation_file = data_location + '/train_captions_psumsq_lions.txt'
#     val_annotation_file  = data_location + '/val_captions_psumsq_lions.txt'
#     test_annotation_file  = data_location + '/test_captions_psumsq_lions.txt'

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
#     profiler = Profiler(async_mode='disabled')
#     profiler.start()

#     print_time('\n ------------------------ \n calling get_mean')
    
#     # 500 x 500 is the size of syntetic data before resize, 100x100 is after resize
#     mean = get_mean(train_dataset, train_loader, 100 , 100)

#     mean_file = f'{model_folder}/mean.txt'
#     with open(mean_file, 'w') as file:
#         file.write(str(float(mean)))

#     print_time(f'finished calculating the mean: {mean} and saved it to file: {mean_file}')

#     profiler.stop()
#     profiler.print()

    # ----------------------------------------- Calc std --------------------------------------------------
#     profiler = Profiler(async_mode='disabled')
#     profiler.start()

#     print_time('\n ------------------------ \n calling get_std')

#     std = get_std(train_dataset, train_loader, mean,100,100)

#     std_file = f'{model_folder}/std.txt'
#     with open(std_file, 'w') as file:
#         file.write(str(float(std)))

#     print_time(f'finished calculating the std: {std} and saved it to file: {std_file}')

#     profiler.stop()
#     profiler.print()

    # ----------------------------------------- expermintal mean/std --------------------------------------------------

    # Calculated those values from '/home/space/datasets/COA/generated-data-api-single/res_images' dataset  
    # Reason is to normlaize real images to match colors of synthtic data
    # expermintal 
    mean,std = (torch.tensor(0.5654), torch.tensor(0.2895))
    print_time(f'Using already calculated mean and std in generated-data-api-single dataset, the mean={mean} and std={mean}')
    # ----------------------------------------- expermintal mean/std --------------------------------------------------

    
    # ---------------------------------------- Loaders ---------------------------------------------------------------
    torch.cuda.empty_cache()

    # Defining the transform to be applied
#     mp.set_start_method('spawn')

    transform = T.Compose([
        T.Resize(226),                     
        T.RandomCrop(224),                 
        T.ToTensor(),                               
        T.Normalize(mean, std),
        Noise(0.1, 0.05)
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
    # encoder_dim=2048  ### resnet50
    encoder_dim=512  ### resnet34 & resnet18
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
    print('Initialize new model, loss etc')
    model, optimizer, criterion = get_new_model(hyper_params, learning_rate, ignored_idx, drop_prob, device, True)
    starting_epoch = 1
    
    # if we want to continue training on real dataset
    if continue_from_checkpoint and real_data and baseline_model:
        model, optimizer, loss, criterion = load_model(model_folder + '/'+ baseline_model,
                                            hyper_params,
                                            learning_rate,
                                            drop_prob,
                                            ignored_idx,
                                            pretrained=True)
    elif continue_from_checkpoint:
            print('Loading model from latest saved checkpoint')   
            model, optimizer, starting_epoch = load_model_checkpoint(model_folder + '/checkpoint.pt', model, optimizer, device)


    # --------------
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 5

    model, train_loss, valid_loss, avg_acc, bleu_score, avg_acc_charge_only, avg_acc_charge_color, avg_acc_shield_only = train_model(
        model, optimizer, criterion, train_dataset, train_loader, val_loader, val_dataset, vocab_size, batch_size, patience, num_epochs, device, model_folder, starting_epoch, weights_map)

    final_accuracy = sum(avg_acc)/len(avg_acc) if len(avg_acc) > 0 else 0.0
    final_train_loss = sum(train_loss)/len(train_loss) if len(avg_acc) > 0 else 0.0
    final_valid_loss = sum(valid_loss)/len(valid_loss) if len(avg_acc) > 0 else 0.0

    final_accuracy_charge_only = sum(avg_acc_charge_only)/len(avg_acc_charge_only) if len(avg_acc_charge_only) > 0 else 0.0
    final_accuracy_charge_color = sum(avg_acc_charge_color)/len(avg_acc_charge_color) if len(avg_acc_charge_color) > 0 else 0.0
    final_accuracy_shield_only = sum(avg_acc_shield_only)/len(avg_acc_shield_only) if len(avg_acc_shield_only) > 0 else 0.0
    
    # print('Bleu Score: ', bleu_score/8091)
    print('Final accuracy ALL: {}%'.format(100. * round(final_accuracy, 2)))
    
    print('Final accuracy Charge-Mod: {}%'.format(100. * round(final_accuracy_charge_only, 2)))
    print('Final accuracy Charge color : {}%'.format(100. * round(final_accuracy_charge_color, 2)))
    print('Final accuracy Shield: {}%'.format(100. * round(final_accuracy_shield_only, 2)))

    print('Final train_loss:  {}'.format(round(final_train_loss, 2)))
    print('Final valid_loss:  {}'.format(round(final_valid_loss, 2)))

    # ---------------------------------------- Saving the model ----------------------------------------------------

    # save the latest model
    now = datetime.now() # current date and time
    timestr = now.strftime("%m-%d-%Y-%H:%M:%S")
    model_full_path = f"{model_folder}/baseline-model-{timestr}.pth"

    save_model(model, optimizer, final_train_loss, final_accuracy, model_full_path, hyper_params)
    print('The trained model has been saved to ', model_full_path)
