#!/usr/bin/python

#imports 
import os
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
from src.baseline.data_loader import get_loader, get_loaders, get_mean_std
from src.accuracy import Accuracy
from src.baseline.coa_model import save_model, get_new_model, validate_model, train_validate_test_split
import torch.multiprocessing as mp
from src.utils import print_time

from pyinstrument import Profiler
from datetime import datetime


if __name__ == "__main__":
    print('starting the script')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    data_location = '/home/space/datasets/COA/generated-data-api'
#     data_location = '/home/space/datasets/COA/generated-data-api-small'

    print('Dataset exists in', data_location)    
    caption_file = data_location + '/captions.txt'
    root_folder_images = data_location + '/images'
    df = pd.read_csv(caption_file)

    train, validate, test = train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None)


    train_annotation_file = data_location + '/train_captions.txt'
    val_annotation_file  = data_location + '/val_captions.txt'
    test_annotation_file  = data_location + '/test_captions.txt'

    train.to_csv(train_annotation_file, sep=',',index=False)
    test.to_csv(test_annotation_file, sep=',',index=False)
    validate.to_csv(val_annotation_file, sep=',',index=False)


    print("There are {} total images".format(len(df)))

    caption_file = data_location + '/train_captions.txt'
    df1 = pd.read_csv(caption_file)
    print("There are {} train images".format(len(df1)))

    caption_file = data_location + '/val_captions.txt'
    df2 = pd.read_csv(caption_file)
    print("There are {} val images".format(len(df2)))

    caption_file = data_location + '/test_captions.txt'
    df3 = pd.read_csv(caption_file)
    print("There are {} test images".format(len(df3)))
    
    # -------------------------------------------------------------------------------------------------------
    
    #setting the constants
    BATCH_SIZE = 125
    NUM_WORKER = 2 #### this needs multi-core
    freq_threshold = 5
    
    # 30 minutes to create those, as it's baseline, i ran it several times and it's the same
    vocab = Vocabulary(freq_threshold)
    vocab.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'g': 4, 'v': 5, 'b': 6, 'cross': 7, 'lion': 8, 'passt': 9, 's': 10, 'a': 11, 'eagle': 12, 'o': 13, 'doubleheaded': 14, "'s": 15, 'head': 16, 'patonce': 17, 'moline': 18, 'guard': 19, 'rampant': 20}
    vocab.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>', 4: 'g', 5: 'v', 6: 'b', 7: 'cross', 8: 'lion', 9: 'passt', 10: 's', 11: 'a', 12: 'eagle', 13: 'o', 14: 'doubleheaded', 15: "'s", 16: 'head', 17: 'patonce', 18: 'moline', 19: 'guard', 20: 'rampant'}
    
    # -------------------------------------------------------------------------------------------------------
    print_time('\n ------------------------ \n before get_loader')
    profiler = Profiler(async_mode='disabled')
    profiler.start()

    train_loader, train_dataset = get_loader(
        root_folder=root_folder_images,
        annotation_file=train_annotation_file,
        transform=None,  # <=======================
        num_workers=NUM_WORKER,
        vocab=vocab,
        batch_size=BATCH_SIZE,
        pin_memory=False
    )
    
    profiler.stop()
    profiler.print()

    profiler = Profiler(async_mode='disabled')
    profiler.start()

    print_time('\n ------------------------ \n before get_mean_std')

    mean, std = get_mean_std(train_dataset, train_loader, 500 , 500)
    print('mean, std:', mean, std)
    profiler.stop()
    profiler.print()
    
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
        batch_size=BATCH_SIZE,
        device=device,
        pin_memory=False
    )

    print_time('finished writing the dataloader')
    # -------------------------------------------------------------------------------------------------------

    #Hyperparams
    embed_size=300
    vocab_size = len(train_dataset.vocab)
    attention_dim=256
    encoder_dim=2048
    decoder_dim=512
    learning_rate = 3e-4
    drop_prob=0.3
    ignored_idx = train_dataset.vocab.stoi["<PAD>"]

    hyper_params = {'embed_size': embed_size,
                    'attention_dim': attention_dim,
                    'encoder_dim': encoder_dim,
                    'decoder_dim': decoder_dim,
                    'vocap_size': vocab_size
                  }
    
    # -------------------------------------------------------------------------------------------------------
    print('initialize new model, loss etc')    
    model, optimizer, criterion = get_new_model(embed_size, vocab_size, attention_dim, encoder_dim,
                                                decoder_dim, learning_rate,drop_prob,ignored_idx, device) 

    losses = list()
    losses_batch = list()
    val_losses = list()
    accuracy_list = list()
    
    now = datetime.now() # current date and time
    timestr = now.strftime("%m.%d.%Y-%H:%M:%S")
    model_full_path = f"/home/space/datasets/COA/models/baseline/attention_model_acc_qsub-{timestr}.pth"
    num_epochs = 5
    print_every = 5

    print('Start Training the model')    
    for epoch in range(1, num_epochs + 1): 
        with tqdm(train_loader, unit="batch") as tepoch:
            idx = 0
            avg_val_loss, bleu_score, accuracy = 0,0,0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)
            for image, captions in tepoch:
                idx+=1
                tepoch.set_description(f"Epoch {epoch}")
                image, captions = image.to(device), captions.to(device)

                # Zero the gradients.
                optimizer.zero_grad()

                # Feed forward
                outputs, attentions = model(image, captions.T)

                # Calculate the batch loss.
                targets = captions.T[:,1:]  ####### the fix in here
                loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

                # Backward pass. 
                loss.backward()

                # Update the parameters in the optimizer.
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                sleep(0.1)

                avg_val_loss, bleu_score, accuracy = validate_model(model, criterion, val_loader, val_dataset, vocab_size, device)
                model.train()

                losses_batch.append(loss) 
                val_losses.append(avg_val_loss)
                accuracy_list.append(accuracy)

            avg_batch_loss = sum(losses_batch) / len(losses_batch)
            losses.append(avg_batch_loss)


    print('Bleu Score: ',bleu_score/8091)
    print('Final accuracy: ', sum(accuracy_list)/len(accuracy_list))

    # save the latest model
    save_model(model, optimizer, epoch, loss, accuracy, model_full_path, hyper_params)
    print('The trained model has been saved to ', model_full_path)
