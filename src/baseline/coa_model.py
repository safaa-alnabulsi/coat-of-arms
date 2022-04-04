import os
import spacy
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx as onnx
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
from time import sleep
from src.baseline.model import EncoderCNN, Attention, DecoderRNN, EncoderDecoder
from src.accuracy import Accuracy
from src.pytorchtools import EarlyStopping, EarlyStoppingAccuracy
from src.utils import list_of_tensors_to_numpy_arr


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


def get_new_model(hyper_params, learning_rate, ignored_idx, drop_prob, device):
    embed_size = hyper_params['embed_size']
    vocab_size = hyper_params['vocab_size']
    encoder_dim = hyper_params['encoder_dim']
    decoder_dim = hyper_params['decoder_dim']
    attention_dim = hyper_params['attention_dim']

    model = EncoderDecoder(embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=ignored_idx)

    return model, optimizer, criterion


# Function to test the model with the val dataset and print the accuracy for the test images
def validate_model(model, criterion, val_loader, val_dataset, vocab_size, device):
#     print('validate function called')
    accuracy_list = list()
    total = len(val_loader)
    bleu_score = 0
    correct = 0
    val_losses = list()
    avg_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, (img, correct_cap) in enumerate(iter(val_loader)):
            features = model.encoder(img.to(device))

            features_tensors = img[0].detach().clone().unsqueeze(0)
            features = model.encoder(features_tensors.to(device))
                        
            caps,_ = model.decoder.generate_caption(features, vocab=val_dataset.vocab)   
            caps = caps[:-1]
            predicted_caption = ' '.join(caps)

            correct_caption = []
            for j in correct_cap.T[0]:
                if j.item() not in [0, 1, 2 , 3]:
                    correct_caption.append(val_dataset.vocab.itos[j.item()])
            correct_caption_s = ' '.join(correct_caption)
            # ------------------------------------------
            # calc metrics

            accuracy_list.append(Accuracy(predicted_caption,correct_caption_s).get())

            bleu = nltk.translate.bleu_score.sentence_bleu([correct_caption], caps, weights=(0.5, 0.5))
            bleu_score += bleu

            # ------------------------------------------
            # calc losses and take the average 
            image, captions = img.to(device), correct_cap.to(device)
            outputs, _ = model(image, captions.T)
            targets    = captions.T[:,1:] 
            loss       = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            val_losses.append(loss)
            
            # ------------------------------------------
           
    # compute the accuracy over all test images
#     acc_score = (100 * sum(accuracy_list) / len(accuracy_list))
#     avg_loss = sum(val_losses) / len(val_losses)
#     print('avg_loss, bleu_score, acc_score', avg_loss, bleu_score, acc_score)

    return val_losses, accuracy_list, bleu_score


def train_model(model, optimizer, criterion, 
                train_loader, val_loader, val_dataset, 
                vocab_size, batch_size, patience, n_epochs, device):

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # to track the accuracy as the model trains
    accuracy_list = []
    avg_acc = []
    
    # initialize the early_stopping object
#     early_stopping = EarlyStopping(patience=patience, verbose=True)
    early_stopping = EarlyStoppingAccuracy(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):
        with tqdm(train_loader, unit="batch") as tepoch:

            ###################
            # train the model #
            ###################
            model.train() # prep model for training
            for image, captions in tepoch: 
                tepoch.set_description(f"Epoch {epoch}")
                # use cuda
                image, captions = image.to(device), captions.to(device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs, attentions = model(image, captions.T)
                # calculate the loss
                targets = captions.T[:,1:]  ####### the fix in here
                loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # record training loss
                train_batch_loss = loss.item()
                train_losses.append(train_batch_loss)
                tepoch.set_postfix({'Train loss (in progress)': train_batch_loss})

               
            ######################    
            # validate the model #
            ######################

            val_losses, accuracy_list, bleu_score = validate_model(model, criterion, 
                                                                   val_loader, val_dataset,
                                                                   vocab_size, device)

            ########################################    
            # print training/validation statistics #
            ########################################

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)

            # Copy the tensor to host memory first to move tensor to numpy
            valid_losses = list_of_tensors_to_numpy_arr(val_losses)        
            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)

            # calculate average accuracy over an epoch       
            accuracy = np.average(accuracy_list)
            avg_acc.append(accuracy)

            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} , ' +
                         f'valid_loss: {valid_loss:.5f} , ' +
                         f'accuracy: {accuracy:.5f}')
        
            print(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            accuracy_list = []

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
    #         early_stopping(valid_loss, model)
            early_stopping(accuracy, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return  model, avg_train_losses, avg_valid_losses, avg_acc, bleu_score

# helper function to save the model 
# https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
def save_model(model, optimizer, loss, accuracy, model_full_path, hyper_params):
    model.cpu()
    model_state = {
        'hyper_params': hyper_params,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }

    torch.save(model_state, model_full_path)

def load_model(model_path, hyper_params, learning_rate, drop_prob, ignored_idx):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, optimizer, criterion = get_new_model(hyper_params, learning_rate, ignored_idx, drop_prob, device)
        
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']

    return model, optimizer, loss,criterion

def load_model_checkpoint(model_path, hyper_params, learning_rate, drop_prob, ignored_idx):    
    model, optimizer, criterion = get_new_model(hyper_params, learning_rate, ignored_idx, drop_prob, device)
        
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss
    
