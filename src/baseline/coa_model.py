from itertools import starmap
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
import torchvision.transforms as T

from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from tqdm import tqdm
from time import sleep

from src.baseline.model import EncoderCNN, Attention, DecoderRNN, EncoderDecoder
from src.baseline.data_loader import get_loader, get_mean, get_std
from src.accuracy import Accuracy
from src.pytorchtools import EarlyStopping, EarlyStoppingAccuracy
from src.utils import list_of_tensors_to_numpy_arr
from datetime import datetime


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


def get_new_model(hyper_params, learning_rate, ignored_idx, drop_prob, device, pretrained):
    embed_size = hyper_params['embed_size']
    vocab_size = hyper_params['vocab_size']
    encoder_dim = hyper_params['encoder_dim']
    decoder_dim = hyper_params['decoder_dim']
    attention_dim = hyper_params['attention_dim']

    model = EncoderDecoder(embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob, pretrained).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=ignored_idx)

    return model, optimizer, criterion


def predict_image(model,image, dataset, device):
    # encode the image to be ready for prediction
    # features = model.encoder(image.to(device))
    features_tensor = image.detach().clone().unsqueeze(0)
    features = model.encoder(features_tensor.to(device))
    
    # predict the caption from the image
    caps,_ = model.decoder.generate_caption(features, vocab=dataset.vocab)   
    caps = caps[:-1]
    predicted_caption = ' '.join(caps)
            
    return  predicted_caption, caps


def get_correct_caption_as_string(dataset, correct_cap):
    correct_caption = []
    for j in correct_cap.T:
        # 0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'
        item = j.item()
        if item not in [0, 1, 2 , 3]:
            itos = dataset.vocab.itos[item]
            correct_caption.append(itos) 
    return ' '.join(correct_caption)


# Function to test the model with the val dataset and print the accuracy for the test images
def validate_model(model, criterion, val_loader, val_dataset, vocab_size, device, tepoch, writer, step):
    print('validate function called')
    total = len(val_loader)
    bleu_score = 0

    accuracies = []
    losses = []

    model.eval()
    with torch.no_grad():
        # loop over batches
        for idx, (imgs, correct_caps,_,_,image_file_names) in enumerate(iter(val_loader)):
            # ------------------------------------------
            # calc losses and take the average 
            images, captions = imgs.to(device), correct_caps.to(device)
            # predicted - forward pass: compute predicted outputs by passing inputs to the model
            outputs, _ = model(images, captions)
            # correct            
            targets    = captions[:,1:] 
            loss       = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            losses.append(loss)
            tepoch.set_postfix({'validatio loss (in progress)': loss})
            
            # loop over images of one batch and predict them one by one to calculate accuracy
            for img, correct_cap,image_file_name in zip(images,correct_caps, image_file_names):
                predicted_caption, caps = predict_image(model, img, val_dataset, device)
                # get the correct caption as a string
                correct_caption_s = get_correct_caption_as_string(val_dataset, correct_cap)
                # ------------------------------------------
                # calc metrics
                print(f'Calc Accuracy image_file_name: {image_file_name}, correct_caption_s: {correct_caption_s}, predicted_caption: {predicted_caption}') 
                try:
                    acc = Accuracy(predicted_caption,correct_caption_s).get()
                except ValueError as e:
                    print(f'Problem in Image {image_file_name}, {correct_caption_s}, {predicted_caption}') 
                    acc = 0.0
                
                accuracies.append(acc)

                # bleu = nltk.translate.bleu_score.sentence_bleu([correct_caption], caps, weights=(0.5, 0.5))
                # bleu_score += bleu

            # ------------------------------------------
    avg_loss = sum(losses)/len(losses)
    avg_acc = sum(accuracies)/len(accuracies)

    writer.add_scalar("Loss/validation", avg_loss, step)
    writer.add_scalar("Accuracy/validation", avg_acc, step)
    
    # compute the accuracy over all test images
#     acc_score = (100 * sum(accuracy_list) / len(accuracy_list))
#     avg_loss = sum(losses) / len(losses)
#     print('avg_loss, bleu_score, acc_score', avg_loss, bleu_score, acc_score)
    writer.close()

    return losses, accuracies, bleu_score, tepoch

def print_time_now(epoch):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"Current Time ={current_time} in epoch {epoch}")

def train_model(model, optimizer, criterion, 
                train_dataset, train_loader, 
                val_loader, val_dataset, 
                vocab_size, batch_size, patience, n_epochs, device, model_folder, starting_epoch=1):

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_loss=0
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # to track the accuracy as the model trains
    accuracy_list = []
    avg_acc = []
    
    # number of batchs is needed to calclualte the validation interval
    num_of_batches = len(train_loader)

    # Writer will store the model training progress
    writer = SummaryWriter(f"{model_folder}/logs/")

    # initialize the early_stopping object
    checkpoint_file=f"{model_folder}/checkpoint.pt"
    # early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_file)
    early_stopping = EarlyStoppingAccuracy(patience=patience, verbose=True, path=checkpoint_file)
    
    loss_idx_value = 0
    
    print(f"Starting training from epoch {starting_epoch}")
    
    for epoch in range(starting_epoch, n_epochs + 1):
        print_time_now(epoch)
        with tqdm(train_loader, unit="batch") as tepoch:
            validation_interval = 0
            ###################
            # train the model #
            ###################
            model.train() # prep model for training
            for batch_images, captions,_,_,_ in tepoch: 
                tepoch.set_description(f"Epoch {epoch}")
                # use cuda
                batch_images, captions = batch_images.to(device), captions.to(device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs, attentions = model(batch_images, captions)
                # calculate the loss
                targets = captions[:,1:]  ####### the fix in here
                loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # record training loss
                train_batch_loss = loss.item()
                train_losses.append(train_batch_loss)
                
                tepoch.set_postfix({'Train loss (in progress)': train_batch_loss})
                writer.add_scalar("Loss/train per batch", train_batch_loss, loss_idx_value)
                
                # # Accuracy of training
                # for img, correct_cap in zip(batch_images, captions):
                #     # nice to have:  training accuracy
                #     predicted_caption,caps = predict_image(model, img, train_dataset, device)
                #     # get the correct caption as a string
                #     correct_caption_s = get_correct_caption_as_string(train_dataset, correct_cap)
                #     acc = Accuracy(predicted_caption,correct_caption_s).get()
                #     accuracy_training_list.append(acc)
                #     writer.add_scalar("Accuracy/train per batch image", acc, loss_idx_value)

                ######################    
                # validate the model #
                ######################
                # validation interval: let's do it 10 times 
                # for 1370 batchs (512 batch size)
                # proof:
                # counter=0
                # for i in range(0,1370):
                #     if i%(int(1370/10)) == 0:
                #         counter+=1
                # print(counter)
                # if validation_interval % (num_of_batches/10) == 0:
                #     val_losses, accuracy_list, bleu_score, tepoch = validate_model(
                #         model,
                #         criterion, 
                #         val_loader,
                #         val_dataset,
                #         vocab_size,
                #         device,
                #         tepoch,
                #         writer,
                #         step=loss_idx_value
                #     )
                    
                loss_idx_value += 1
                # validation_interval+=1

            ########################################    
            # print training/validation statistics #
            ########################################

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)
            writer.add_scalar("Loss/train per epoch", train_loss, epoch)

            # Copy the tensor to host memory first to move tensor to numpy
            # notice in here you are getting only the latest validation values
            # valid_losses = list_of_tensors_to_numpy_arr(val_losses)        
            # valid_loss = np.average(valid_losses)
            # avg_valid_losses.append(valid_loss)

            # calculate average accuracy over an epoch       
            accuracy = np.average(accuracy_list)
            avg_acc.append(accuracy)
            writer.add_scalar("Accuracy/train per epoch", accuracy, epoch)

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
    #         early_stopping(valid_loss, model, optimizer)
            early_stopping(accuracy, model, optimizer, epoch)

            if early_stopping.early_stop:
                print("Early stopping. Stopping the training of the model.")
                break
            
        val_losses, accuracy_list, bleu_score, tepoch = validate_model(
                        model,
                        criterion, 
                        val_loader,
                        val_dataset,
                        vocab_size,
                        device,
                        tepoch,
                        writer,
                        step=epoch
                    )
        
        valid_losses = list_of_tensors_to_numpy_arr(val_losses)        
        valid_loss = np.average(valid_losses)
        avg_valid_losses.append(valid_loss)
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} , ' +
                        f'valid_loss: {valid_loss:.5f} , ' +
                        f'accuracy: {accuracy:.5f}')

        print(print_msg)

    writer.close()

    # load the last checkpoint with the best model
    model, _, _ = load_model_checkpoint(checkpoint_file, model, optimizer, device)
    
    return  model, avg_train_losses, avg_valid_losses, avg_acc, bleu_score

# ---------- model save/load function ------------ #

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

def load_model(model_path, hyper_params, learning_rate, drop_prob, ignored_idx, pretrained):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, optimizer, criterion = get_new_model(hyper_params, learning_rate, ignored_idx, drop_prob, device, pretrained)
        
    checkpoint = torch.load(model_path,map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']

    return model, optimizer, loss,criterion

# ---------- checkpoint save/load function ------------ #

def load_model_checkpoint(checkpoint_file, model, optimizer, device):    
    checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
        
    return model, optimizer, epoch

# ---------- testing model function ------------ #

def get_training_mean_std(run_path):
    f = open(run_path+"/mean.txt", "r")
    mean = torch.tensor(float(f.read()), dtype=torch.float)

    f = open(run_path+"/std.txt", "r")
    std = torch.tensor(float(f.read()), dtype=torch.float)
    print(f'mean={mean}, std={std}')

    return mean, std

def init_testing_model(test_caption_file, root_folder_images, mean, std,
                       num_worker,vocab,batch_size, device, pin_memory=False,img_h=500, img_w=500):
    test_loader, test_dataset = get_loader(
        root_folder=root_folder_images,
        annotation_file=test_caption_file,
        transform=None,  # <=======================
        num_workers=num_worker,
        vocab=vocab,
        batch_size=batch_size,
        device=device,
        pin_memory=pin_memory
    )

    #defining the transform to be applied

    transform = T.Compose([
        T.Resize(226),                     
        T.RandomCrop(224),                 
        T.ToTensor(),                               
        T.Normalize(mean, std) 
    ])

    test_loader, test_dataset = get_loader(
        root_folder=root_folder_images,
        annotation_file=test_caption_file,
        transform=transform,  # <=======================
        num_workers=num_worker,
        vocab=vocab,
        batch_size=batch_size,
        device=device,
        pin_memory=pin_memory
    )
    
    return test_loader, test_dataset

def test_model(model, criterion, test_loader, test_dataset, vocab_size, device, model_folder, real_data):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    test_losses=[]
    accuracy_test_list=[] # to calculate average accuracy per all batches, only one value per batch
    accuracy_batch_list=[] # to calculate average accuracy per all images in one batch, re-emptied every batch
    accuracy_all_images_list=[] # to save all values of accuracy of all images to be used in get_min_max_acc_images function
    image_names_list=[]

    # Writer will store the model test results progress
    writer = SummaryWriter(f"{model_folder}/logs")

    # adjust scalar name based on real data or test data
    scalar_test =  'Synthetic data'
    if real_data:
        scalar_test = 'Real data'

    model.eval()
    with torch.no_grad():
        for idx, (batch_images, correct_cap,_,_,image_file_names) in enumerate(iter(test_loader)):
            # print(f'test function len batch_images {len(batch_images)} in batch {idx}')
            # print(f'test function image_file_names {image_file_names} in batch {idx}')
            # print(f'test function correct_cap {correct_cap} in batch {idx}')

            # calc losses and take the average 
            images, captions = batch_images.to(device), correct_cap.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs, _ = model(images, captions)
            targets    = captions[:,1:] 
            # calculate the loss
            loss       = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            test_losses.append(loss)
           
            # TensorBoard scalars
            writer.add_scalar(f"Loss/test {scalar_test}", loss, idx)
            # ------------------------------------------
            accuracy_batch_list=[]
            for img, correct_cap, image_file_name in zip(batch_images, correct_cap, image_file_names):
            
                predicted_caption,_ = predict_image(model, img, test_dataset, device)
                
                # get the correct caption as a string
                correct_caption_s = get_correct_caption_as_string(test_dataset, correct_cap)

                # calc metrics
                acc_image = Accuracy(predicted_caption, correct_caption_s).get()
                accuracy_batch_list.append(acc_image)
                
                # to be used later in get_min_max_acc_images function
                accuracy_all_images_list.append(acc_image)
                image_names_list.append(image_file_name)

            avg_batch_acc = sum(accuracy_batch_list)/len(accuracy_batch_list)
            accuracy_test_list.append(avg_batch_acc)
            
            test_batch_loss_per = 100. * avg_batch_acc
            print(f'Test Acuuracy (in progress): {test_batch_loss_per}%')
            print("--------------------------")
            writer.add_scalar(f"Accuracy/test {scalar_test}", avg_batch_acc, idx)

    # calculate and print avg test loss
    test_loss = sum(test_losses)/len(test_dataset)
    test_loss_per = 100. * test_loss
    
    print(f'Test Loss (final): {test_loss_per}%')

    acc_test_score = (100. * sum(accuracy_test_list) / len(accuracy_test_list))

    print(f'Test Accuracy (Overall): {acc_test_score}%')
    
    return test_losses, accuracy_test_list, image_names_list, acc_test_score, test_loss


def get_min_max_acc_images(accuracy_test_list, image_names_list):
    """Get the image names of both min and max accuracies inserted together
        TODO: change it later to top 5 of each category

    Args:
        accuracy_test_list ([float]): list of accuracies of all images
        image_names_list ([string]): list of image file names

    Returns:
        [string]: lowest_acc, list of 5 image file name with lowest accuracy 
        [string]: highest_acc, list of 5 image file name with highest accuracy
    """
    highest_acc = []
    lowest_acc = []
    sorted_idx_accuracy = np.argsort(accuracy_test_list)
    for i in image_names_list[:5]:
        lowest_acc.append(i)
    for i in image_names_list[-5:]:
        highest_acc.append(i)

    return lowest_acc, highest_acc
    # code here to get only two images, min and max 
    # idx_image_with_max_acc = accuracy_test_list.index(max(accuracy_test_list))
    # idx_image_with_min_acc = accuracy_test_list.index(min(accuracy_test_list))

    # image_with_max_acc = image_names_list[idx_image_with_max_acc]
    # image_with_min_acc = image_names_list[idx_image_with_min_acc]
    
    # return image_with_max_acc, image_with_min_acc

##  Visualizing the attentions
# Defining helper functions
# Given the image generate captions and attention scores</li>
# Plot the attention scores in the image</li>

# generate caption
def get_caps_from(model, test_dataset, features_tensors, device):
    #generate the caption
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps,alphas = model.decoder.generate_caption(features,vocab=test_dataset.vocab)
        caption = ' '.join(caps)
        show_image(features_tensors[0],title=caption)
    
    return caps,alphas

#Show attention
def plot_attention(img, result, attention_plot):
    #untransform
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    for l in range(len_result):
        temp_att = attention_plot[l].reshape(7,7)
        
        ax = fig.add_subplot(len_result//2,len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())
        

    plt.tight_layout()
    plt.show()
    
def test_rand_image(model, test_dataset, test_loader,device):
    dataiter = iter(test_loader)
    images,_,_,_ = next(dataiter)

    img = images[0].detach().clone()
    img1 = images[0].detach().clone()
    caps,alphas = get_caps_from(model, test_dataset, img.unsqueeze(0),device)

    plot_attention(img1, caps, alphas)

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
