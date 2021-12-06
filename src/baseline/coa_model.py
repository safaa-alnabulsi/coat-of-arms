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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from time import sleep
from src.baseline.model import EncoderCNN, Attention, DecoderRNN, EncoderDecoder
from src.accuracy import Accuracy


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


def get_new_model(embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, learning_rate,drop_prob,ignored_idx, device):
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
    acc_score = (100 * sum(accuracy_list) / len(accuracy_list))
    avg_loss = sum(val_losses) / len(val_losses)
#     print('avg_loss, bleu_score, acc_score', avg_loss, bleu_score, acc_score)

    return avg_loss, bleu_score, acc_score


#helper function to save the model
def save_model(model, optimizer, epoch, loss, accuracy, model_full_path, hyper_params):
    model.cpu()
    model_state = {
        'epoch': epoch,
        'hyper_params': hyper_params,
        'state_dict': model.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy':accuracy
    }
    
    torch.save(model_state, model_full_path)

def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model, _, _ = get_new_model()
    
    model.load_state_dict(torch.load(model_path))

    return model

def load_model_checkpoint(model_path):
    checkpoint = torch.load(model_path)
    
    model, optimizer, criterion = get_new_model()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss

#show the tensor image
def show_image(img, title=None):
    """Imshow for Tensor."""
    
    #unnormalize 
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
      
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    

def print_time(text):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("{} @ Time = {}".format(text, current_time))
    
    


# def train_model(model, optimizer, criterion, train_loader, train_dataset, 
#                 val_loader, val_dataset, 
#                 num_epochs, vocab_size,
#                 hyper_params,
#                 device, model_full_path):
#     losses = list()
#     losses_batch = list()
#     val_losses = list()
#     val_accuracy_list = list()

#     # if model is None:
    
    
# #     train_on_gpu = torch.cuda.is_available()

# #     if train_on_gpu:
# #         print("CUDA is available! Training on GPU...")
# #     else:
# #         print("CUDA is not available. Training on CPU...")
    
# #     # Move tensors to GPU is CUDA is available
# #     if train_on_gpu:
# #         model.cuda()

#     for epoch in range(1, num_epochs + 1): 
#     #     if model is None:
#     #         model, optimizer, epoch, loss = load_model_checkpoint(model_full_path)

#         with tqdm(train_loader, unit="batch") as tepoch:
#             idx = 0
#             for image, captions in tepoch:
#                 idx+=1
#                 tepoch.set_description(f"Epoch {epoch}")
#                 image, captions = image.to(device), captions.to(device)

#                 # Zero the gradients.
#                 optimizer.zero_grad()

#                 # Feed forward
#                 outputs, attentions = model(image, captions.T)

#                 # Calculate the batch loss.
#                 targets = captions.T[:,1:]  ####### the fix in here
#                 loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

#                 # Backward pass. 
#                 loss.backward()

#                 # Update the parameters in the optimizer.
#                 optimizer.step()

#                 tepoch.set_postfix(loss=loss.item())
#                 sleep(0.1)

#                 avg_val_loss, bleu_score, accuracy = validate_model(model, criterion, 
#                                                                     val_loader, val_dataset, 
#                                                                     vocab_size, device)
#                 model.train()

#                 losses_batch.append(loss) # in here 17 batches * 5 epochs = 85 , you can get the average
#                 val_losses.append(avg_val_loss)
#                 val_accuracy_list.append(accuracy)

#             avg_batch_loss = sum(losses_batch) / len(losses_batch)
#             losses.append(avg_batch_loss)
    
                
#         # save the latest model
#         accuracy = sum(val_accuracy_list)/len(val_accuracy_list)
#         save_model(model, optimizer, epoch, loss, accuracy, model_full_path, hyper_params)

#     return model, losses, val_accuracy_list, val_losses, bleu_score
