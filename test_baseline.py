#!/usr/bin/python

# imports 
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx as onnx
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T

from src.baseline.vocabulary import Vocabulary
from src.utils import print_time, list_of_tensors_to_numpy_arr, plot_image, plot_im
from src.accuracy import WEIGHT_MAP, WEIGHT_MAP_ONLY_SHIELD_COLOR, WEIGHT_MAP_ONLY_CHARGE, WEIGHT_MAP_ONLY_CHARGE_COLOR
from src.baseline.coa_model import get_new_model,load_model, train_validate_test_split, init_testing_model, test_model, test_rand_image, get_training_mean_std, get_min_max_acc_images, calc_acc, calc_acc_on_loader, calc_acc_on_loader_color_only, calc_acc_color_only
from pyinstrument import Profiler
from src.baseline.data_loader import get_loader

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
    parser.add_argument('--caption-file', dest='caption_file', type=str, help='caption file for test images', default='test_captions_psumsq.txt')
    parser.add_argument('--resized-images', dest='resized_images', type=str, help='smaller resized images?', default='no', choices=['yes','Yes','y','Y','no','No','n', 'N'])
    parser.add_argument('--accuracy', dest='accuracy', type=str, help='type of accuracy', default='all', choices=['all','charge-mod-only','charge-color-only','shield-color-only'])
    parser.add_argument('--seed', dest='seed', type=int, help='reproducibility seed', default=1234)
    parser.add_argument('--height', dest='height', type=int, help='height of image', default=100)
    
    args = parser.parse_args()

    # ---------------------- Reproducibility -------------------
    
    seed = args.seed
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # ----------------------------------------------------------------- 

    data_location = args.dataset
    run_name = args.run_name
    model_name = args.model_name
    batch_size = args.batch_size
    local = args.local
    real_data = args.real_data
    resized_images = args.resized_images
    caption_file = args.caption_file
    accuracy = args.accuracy
    height = width = height_synth = int(args.height)
    
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
        # resized to avg size
        height = 634
        width = 621
    else: 
        real_data = False
        
    if resized_images in ['yes','Yes','y','Y'] :
        resized_images = True
    else: 
        resized_images = False
        
    # choices=['all','charge-mod-only','charge-color-only','shield-color-only']  
    weights_map = WEIGHT_MAP           
    if accuracy == 'charge-mod-only':
        weights_map = WEIGHT_MAP_ONLY_CHARGE
    elif accuracy == 'charge-color-only':
        weights_map = WEIGHT_MAP_ONLY_CHARGE_COLOR
    elif accuracy == 'shield-color-only':
        weights_map = WEIGHT_MAP_ONLY_SHIELD_COLOR        

    if real_data==True and resized_images==True:
        height = 634
        width = 621
#         height = 100
#         width = 100
    elif real_data==False and resized_images==True:
            height = height
            width = width
    elif real_data==False and resized_images==False:
            height = 500
            width = 500
        
    print('testing cropped real dataset? ',real_data)
    print('running on local ',local)
    print('data_location is ',data_location)
    print('model_path is ',model_path)
    print('batch_size is ',batch_size)
    print(f'accuracy is {accuracy} and the weights_map is {weights_map}')
    print(f'height: {height}, width: {width}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # --------------------------------------- test dataset ---------------------------------
    
    print('Dataset exists in', data_location)  
    
    test_caption_file  = data_location + '/' + caption_file
    print('Test caption file path: ', test_caption_file)    
  
    if real_data:
        root_folder_images = data_location + '/resized'
    else:
        if resized_images:
            root_folder_images = data_location + f'/res_images{height}x{width}'            
        else:
            root_folder_images = data_location + '/images'
    

    df = pd.read_csv(test_caption_file)
    print("There are {} test images".format(len(df)))

    # ----------------------------------------- Constants -----------------------------------------
    
    #setting the constants
    NUM_WORKER = 2 #### this needs multi-core
    freq_threshold = 5
    
    # 30 minutes to create those, as it's baseline, i ran it several times and it's the same
    vocab = Vocabulary(freq_threshold)
#     vocab.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'lion': 4, 'rampant': 5, 'passt': 6, 'guard': 7, 'head': 8, 'lions': 9, 'cross': 10, 'molxine': 11, 'patonce': 12, 'eagle': 13, 'doubleheaded': 14, 'eagles': 15, 'a': 16, 'b': 17, 'o': 18, 's': 19, 'g': 20, 'e': 21, 'v': 22, '1': 23, '2': 24, '3': 25, '4': 26, '5': 27, '6': 28, '7': 29, '8': 30, '9': 31, '10': 32, '11': 33, 'border': 34, '&': 35}
#     vocab.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>', 4: 'lion', 5: 'rampant', 6: 'passt', 7: 'guard', 8: 'head', 9: 'lions', 10: 'cross', 11: 'moline', 12: 'patonce', 13: 'eagle', 14: 'doubleheaded', 15: 'eagles', 16: 'a', 17: 'b', 18: 'o', 19: 's', 20: 'g', 21: 'e', 22: 'v', 23: '1', 24: '2', 25: '3', 26: '4', 27: '5', 28: '6', 29: '7', 30: '8', 31: '9', 32: '10', 33: '11', 34: 'border', 35: '&'}
    
    # after removing border
    vocab.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'lion': 4, 'rampant': 5, 'passt': 6, 'guard': 7, 'head': 8, 'lions': 9, 'cross': 10, 'moline': 11, 'patonce': 12, 'eagle': 13, 'doubleheaded': 14, 'eagles': 15, 'a': 16, 'b': 17, 'o': 18, 's': 19, 'g': 20, 'e': 21, 'v': 22, '1': 23, '2': 24, '3': 25, '4': 26, '5': 27, '6': 28, '7': 29, '8': 30, '9': 31, '10': 32, '11': 33}
    vocab.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>', 4: 'lion', 5: 'rampant', 6: 'passt', 7: 'guard', 8: 'head', 9: 'lions', 10: 'cross', 11: 'moline', 12: 'patonce', 13: 'eagle', 14: 'doubleheaded', 15: 'eagles', 16: 'a', 17: 'b', 18: 'o', 19: 's', 20: 'g', 21: 'e', 22: 'v', 23: '1', 24: '2', 25: '3', 26: '4', 27: '5', 28: '6', 29: '7', 30: '8', 31: '9', 32: '10', 33: '11'}

#     vocab.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'a': 4, 'b': 5, 'o': 6, 's': 7, 'g': 8, 'e': 9, 'v': 10}
#     vocab.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>', 4: 'a', 5: 'b', 6: 'o', 7: 's', 8: 'g', 9: 'e', 10: 'v'}

    # ------------------------------------------ Get mean & std ---------------------------------------
#     mean, std = get_training_mean_std(run_path)

#     print_time(f'Using calculated mean and std from real dataset, the mean={mean} and std={mean}')

    # ----------------------------------------- expermintal mean/std --------------------------------------------------

    # Calculated those values from '/home/space/datasets/COA/generated-data-api-single/res_images' dataset  
    # Reason is to normlaize real images to match colors of synthtic data
    # expermintal 

#     if height_synth == 100:
#         # /home/space/datasets/COA/generated-single-simple - run-04-14-2023-18:45:41 - resized to 100x100
#     mean,std = (torch.tensor(0.300), torch.tensor(0.3484))
    
#     if height_synth == 200:
#         #/home/space/datasets/COA/generated-single-simple - run-04-14-2023-19:48:02 - resized to 200x200
#         mean,std = (torch.tensor(0.28908), torch.tensor(0.346993))
    
#     if height_synth == 300:
#         #/home/space/datasets/COA/generated-single-simple - run-04-14-2023-19:08:56 - resized to 300x300
#         mean,std = (torch.tensor(0.28852), torch.tensor(0.34906))

#     if height_synth == 400:
#         #/home/space/datasets/COA/generated-single-simple - run-04-14-2023-19:51:31 - resized to 400x400
#         mean,std = (torch.tensor(0.286567), torch.tensor(0.3505))
    
#     if height_synth==500:
#         #/home/space/datasets/COA/generated-single-simple - run-04-14-2023-20:02:07 - not resized/original 500x500
#         mean,std = (torch.tensor(0.285479), torch.tensor(0.35180))

#     print_time(f'Using already calculated mean and std in generated-single-simple dataset, the mean={mean} and std={mean}')
    # ----------------------------------------- expermintal mean/std --------------------------------------------------
    # (621, 634)
    mean,std = (torch.tensor(0.28803130984306335), torch.tensor(0.3481476306915283))
    print_time(f'Using already calculated mean and std in generated-data-api-single dataset, the mean={mean} and std={mean}')

    # ---------------------------------------- Loaders ------------------------------------------------
    test_loader, test_dataset = init_testing_model(test_caption_file, 
                                                   root_folder_images, 
                                                   mean, std,
                                                   NUM_WORKER,
                                                   vocab,
                                                   batch_size, 
                                                   device, 
                                                   pin_memory=False,
                                                   img_h=height, 
                                                   img_w=width)
    
    # ----------------------------------------- Hyperparams ---------------------------------

    # Hyperparams

    embed_size=300
    vocab_size = len(test_dataset.vocab)
    attention_dim=256
#     encoder_dim=2048  ### resnet50
    encoder_dim=512  ### resnet34 & resnet18
    decoder_dim=512
    learning_rate = 0.0009 # 3e-4
    drop_prob=0.5
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
    
    test_losses, accuracy_test_list, image_names_list, predictions_list, acc_test_score, test_loss = test_model(model, 
                                                                            criterion,
                                                                            test_loader, 
                                                                            test_dataset, 
                                                                            vocab_size, 
                                                                            device,
                                                                            run_path,
                                                                            real_data,
                                                                            weights_map)    
    
    # -------------------------------- Saving the results ----------------------------------------
    
    image_with_min_acc, image_with_max_acc = get_min_max_acc_images(accuracy_test_list, image_names_list, predictions_list)
    print('images with highest accuracy', image_with_max_acc)
    print('images with lowest accuracy', image_with_min_acc)
    
    # -------------------------------- Training Accuracy ------------------------------------------

    # resized 
#     data_location = '/home/space/datasets/COA/generated-single-simple/'
#     train_annotation_file = data_location + f'/train_captions_psumsq.txt{height_synth}x{height_synth}'
#     root_folder_images = data_location + f'/res_images{height_synth}x{height_synth}'
    
# original
#     train_annotation_file = data_location + f'/train_captions_psumsq.txt'
#     root_folder_images = data_location + f'/images'            

# real data
    train_annotation_file = data_location + f'/train_captions_psumsq.txt'
    root_folder_images = data_location + f'/resized'            
    
#     train_annotation_file = data_location + f'/resized-txt-files-100x100/train_captions_psumsq.txt'
#     root_folder_images = data_location + f'/resized-images-100x100'            
    
    # only color
#     train_annotation_file = data_location + f'/train_captions_psumsq.txt{height_synth}x{height_synth}-only-color'


    transform = T.Compose([
        T.Resize((height,width)),
        T.ToTensor(),                               
        T.Normalize(mean, std) 
    ])

    train_loader, train_dataset = get_loader(
        root_folder=root_folder_images,
        annotation_file=train_annotation_file,
        transform=transform,  # <=======================
        num_workers=NUM_WORKER,
        vocab=vocab,
        batch_size=batch_size,
        device=device,
        pin_memory=False
    )
    
    print("Calculating training accuracy")    
    calc_acc_on_loader(model, train_loader, train_dataset, device, weights_map, strtoprint="Training")
#     calc_acc_on_loader_color_only(model, train_loader, train_dataset, device, strtoprint="Training")
    