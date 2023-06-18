#!/usr/bin/python

#imports 

import random
import os
import argparse
import time
import numpy as np
import pandas as pd
from PIL import Image
from src.baseline.vocabulary import Vocabulary
import torchdatasets as td
from src.utils import print_time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler

import torchvision.transforms as T
import torchvision.models as models
from torchvision.utils import make_grid
from torch.utils.data import RandomSampler

from matplotlib import pyplot as plt

from src.caption import Caption
from src.baseline.coa_model import save_model, load_model, train_validate_test_split
from src.baseline.data_loader import get_mean, get_std
from torch.utils.data import DataLoader, Dataset
from src.pytorchtools import EarlyStopping, EarlyStoppingAccuracy
from src.baseline.noise import Noise
from torch.utils.data import RandomSampler
import argparse


def calc_accuracy(true, pred):
    accuracies = []
    accuracies_charge_only = []
    accuracies_charge_color = []
    accuracies_shield_only = []

    pred = F.softmax(pred, dim = 1)
    true = torch.zeros(pred.shape[0], pred.shape[1]).scatter_(1, true.unsqueeze(1), 1.)
    batch_size = len(pred)
    
    pred_t, true_t = pred.argmax(-1), true.argmax(-1)
    for i in range(0, batch_size-1):
        score, charge_score, charge_color_score, shield_color_score = calc_predicted_label_accuracy(true_t[i], pred_t[i])
        accuracies.append(score)
        accuracies_charge_only.append(charge_score)
        accuracies_charge_color.append(charge_color_score)
        accuracies_shield_only.append(shield_color_score)
        
    return accuracies, accuracies_charge_only, accuracies_charge_color, accuracies_shield_only

def calc_predicted_label_accuracy(true_class, predicted_class):
    
    true_label = CLASSES_MAP.get(int(true_class))

    charge = true_label[0]
    modifier = true_label[1]
    charge_color = true_label[2]
    shield_color = true_label[3]

    predicted_label = CLASSES_MAP.get(int(predicted_class))

    pre_charge = predicted_label[0]
    pre_modifier = predicted_label[1]
    pre_charge_color = predicted_label[2]
    pre_shield_color = predicted_label[3]
    
    charge_hits = 0
    modifier_hits = 0
    charge_color_score = 0
    shield_color_score = 0
    
    if charge == pre_charge:
        charge_hits+=1
    
    if modifier == pre_modifier:
        modifier_hits+=1

    if charge_color == pre_charge_color:
        charge_color_score+=1
        
    if shield_color == pre_shield_color:
        shield_color_score+=1

    charge_score = (charge_hits + modifier_hits) / 2
    
    score = (charge_score + charge_color_score + shield_color_score) / 3
    
    return score, charge_score, charge_color_score, shield_color_score

def calc_accuracy_standard(true, pred):
    pred = F.softmax(pred, dim = 1)
    true = torch.zeros(pred.shape[0], pred.shape[1]).scatter_(1, true.unsqueeze(1), 1.)
    acc = (true.argmax(-1) == pred.argmax(-1)).float().detach().numpy()
    acc = float((100 * acc.sum()) / len(acc))
    return round(acc, 4)

### Training Code
def train_classification_model(model, optimizer, criterion, lr_scheduler, epochs, train_data_loader, val_data_loader, device):
    from tqdm import tqdm
    
    # initialize the early_stopping object
    checkpoint_file=f"{data_location}/classification-checkpoint.pt"
    early_stopping = EarlyStoppingAccuracy(patience=10, verbose=True, path=checkpoint_file)

    accuracies = []
    accuracies_charge_only = []
    accuracies_charge_color = []
    accuracies_shield_only = []

    avg_acc_ls = []
    avg_acc_ls_charge_only = []
    avg_acc_ls_charge_color = []
    avg_acc_ls_shield_only = []

    avg_train_acc_list = []
    for epoch in range(epochs):
        start = time.time()

        #Epoch Loss & Accuracy
        train_epoch_losses = []
        train_epoch_accuracies = []
        _iter = 1

        #Val Loss & Accuracy
        val_epoch_losses = []
        val_epoch_accuracies = []
        train_epoch_accuracies = []
        train_epoch_losses = []
        # Training
        with tqdm(train_data_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
            for images, labels,_,_,_ in tepoch: 
                images = images.to(device)
                labels = labels.to(device)

                #Reset Grads
                optimizer.zero_grad()

                #Forward ->
                preds = model(images)

#                 #Calculate Accuracy
#                 print("train_classification_model")
                acc = calc_accuracy_standard(labels.cpu(), preds.cpu())
#                 print("-----------------------------------")

                #Calculate Loss & Backward, Update Weights (Step)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
#                 lr_scheduler.step()
                
                #Append loss & acc
                loss_value = loss.item()
                train_epoch_losses.append(loss_value)
                train_epoch_accuracies.append(acc)
                
                tepoch.set_postfix({'train_batch_loss': loss_value})
                tepoch.set_postfix({'train_batch_accuracy': acc})

                if _iter % 500 == 0:
                    print("> Iteration {} < ".format(_iter))
                    print("Iter Loss = {}".format(round(loss_value, 4)))
                    print("Iter Accuracy = {} % \n".format(acc))

                _iter += 1
                
        
        train_epoch_loss = sum(train_epoch_losses) / len(train_epoch_losses)
        train_epoch_accuracy = sum(train_epoch_accuracies) / len(train_epoch_accuracies)
        tepoch.set_postfix({'train_epoch_loss': train_epoch_loss})
        tepoch.set_postfix({'train_epoch_accuracy': train_epoch_accuracy})

        avg_train_acc_list.append(train_epoch_accuracy)

        # tryinf to free the memory
#         gc.collect()
#         torch.cuda.empty_cache()
#         gpu_usage()                             

        #Validation
        with tqdm(val_data_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for images, labels,_,_,images_names in tepoch:
                images = images.to(device)
                labels = labels.to(device)

                #Forward ->
                preds = model(images)

                #Calculate Accuracy
#                 print("evaluation_classification")
#                 print('images_names', images_names)
                
                acc1, acc2, acc3, acc4 = calc_accuracy(labels.cpu(), preds.cpu())
                for i in acc1:
                    accuracies.append(i)
                for i in acc2:
                    accuracies_charge_only.append(i)
                for i in acc3:
                    accuracies_charge_color.append(i)
                for i in acc4:
                    accuracies_shield_only.append(i)

                print("-----------------------------------")

                #Calculate Loss
                loss = criterion(preds, labels)

                #Append loss & acc
                loss_value = loss.item()
                tepoch.set_postfix({'val_epoch_loss': loss_value})
                val_epoch_losses.append(loss_value)

        # ------------------------------------------
        # calc avg values coming out of every batch in one epoch 
        val_epoch_accuracy = sum(accuracies) / len(accuracies)
        
        # breakdown of accuracies - average per validation
        avg_acc_charge_only  = sum(accuracies_charge_only) / len(accuracies_charge_only)
        avg_acc_charge_color = sum(accuracies_charge_color) / len(accuracies_charge_color)
        avg_acc_shield_only  = sum(accuracies_shield_only) / len(accuracies_shield_only)
       
        val_epoch_accuracies.append(val_epoch_accuracy)
        tepoch.set_postfix({'val_epoch_accuracy': val_epoch_accuracy})
        # ------------------------------------------

        train_epoch_loss = np.mean(train_epoch_losses)
        val_epoch_loss = np.mean(val_epoch_losses)
        
        # collect avg loss values coming out of every batch in one epoch
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)

        # ------------------------------------------  
        # collect avg values coming out of every epochs 
        avg_acc_ls.append(val_epoch_accuracy)
        avg_acc_ls_charge_only.append(avg_acc_charge_only)
        avg_acc_ls_charge_color.append(avg_acc_charge_color)
        avg_acc_ls_shield_only.append(avg_acc_shield_only)
        # ------------------------------------------
        end = time.time()

        # Print Epoch Statistics
        print("** Epoch {} ** - Epoch Time {}".format(epoch, int(end-start)))
        print("Train Loss = {}".format(round(train_epoch_loss, 4)))
        print("Train Accuracy = {} % \n".format(train_epoch_accuracy))
        print("Val Loss = {}".format(round(val_epoch_loss, 4)))
        print("Val epoch Accuracy = {} % \n".format(val_epoch_accuracy))
        print("Val Charge Accuracy = {} % \n".format(avg_acc_charge_only))
        print("Val Charge Color Accuracy = {} %\n".format(avg_acc_charge_color))
        print("Val Shield Color Accuracy = {} % \n".format(avg_acc_shield_only))
        
        early_stopping(val_epoch_accuracy, model, optimizer, epoch)
           
        if early_stopping.early_stop:
            print("Early stopping. Stopping the training of the model.")
            break
        print("------------------------------------------------------------")

    val_acc_score = sum(avg_acc_ls) / len(avg_acc_ls)
    acc_score_charge = sum(avg_acc_ls_charge_only) / len(avg_acc_ls_charge_only)
    acc_score_charge_color = sum(avg_acc_ls_charge_color) / len(avg_acc_ls_charge_color)
    acc_score_shield = sum(avg_acc_ls_shield_only) / len(avg_acc_ls_shield_only)
    ftrain_loss = sum(train_loss) / len(train_loss) 
    fval_loss = sum(val_loss) / len(val_loss) 
    train_acc_standard = sum(avg_train_acc_list) / len(avg_train_acc_list) 
    
    print('Final Accuracy ALL (Overall): {}%'.format(100. * round(val_acc_score, 2)))
    print('Final Accuracy Charge-Mod only (Overall): {}%'.format(100. * round(acc_score_charge, 2)))
    print('Final Accuracy Charge color (Overall): {}%'.format(100. * round(acc_score_charge_color, 2)))
    print('Final Accuracy Shield color (Overall): {}%'.format(100. * round(acc_score_shield, 2)))
    print('Final train loss (Overall): {}'.format(round(ftrain_loss, 2)))
    print('Final val loss (Overall): {}'.format(round(fval_loss, 2)))
            
    return model, ftrain_loss, train_acc_standard, fval_loss, val_acc_score

def test_classification_model(model, test_data_loader):
    test_epoch_loss = []
    test_epoch_accuracy = []

    test_loss = []
    test_accuracy = []

    accuracies = []
    accuracies_charge_only = []
    accuracies_charge_color = []
    accuracies_shield_only = []

    accuracy_test_list = []
    accuracy_test_list_charge=[]
    accuracy_test_list_charge_color=[]
    accuracy_test_list_shield=[]

    # model.eval()
    with torch.no_grad():
        for images, labels,_,_,_ in test_data_loader:
            print(type(labels))
            images = images.to("cpu")
            labels = labels.to("cpu")

            #Forward ->
            preds = model(images)

            #Calculate Accuracy
            acc = calc_accuracy_standard(labels.cpu(), preds.cpu())
            acc1,acc2,acc3,acc4 = calc_accuracy(labels.cpu(), preds.cpu())
            for i in acc1:
                accuracies.append(i)
            for i in acc2:
                accuracies_charge_only.append(i)
            for i in acc3:
                accuracies_charge_color.append(i)
            for i in acc4:
                accuracies_shield_only.append(i)

            #Calculate Loss
            loss = criterion(preds, labels)

            #Append loss & acc
            loss_value = loss.item()
            test_epoch_loss.append(loss_value)
            test_epoch_accuracy.append(acc)

            avg_batch_acc = sum(accuracies)/len(accuracies)
            avg_batch_acc_charge = sum(accuracies_charge_only)/len(accuracies_charge_only)
            avg_batch_acc_chrage_color = sum(accuracies_charge_color)/len(accuracies_charge_color)
            avg_batch_acc_shield = sum(accuracies_shield_only)/len(accuracies_shield_only)
            
#             avg_batch_acc_charge, avg_batch_acc_chrage_color, avg_batch_acc_shield = 0,0,0
            accuracy_test_list.append(avg_batch_acc)
            accuracy_test_list_charge.append(avg_batch_acc_charge)
            accuracy_test_list_charge_color.append(avg_batch_acc_chrage_color)
            accuracy_test_list_shield.append(avg_batch_acc_shield)
            
            print("Test Accuracy Standard (in progress) = {} % \n".format(acc))
            print('Test Accuracy ALL (in progress): {}%'.format(100. * round(avg_batch_acc, 2)))
            print('Test Accuracy Charge-Mod only (in progress): {}%'.format(100. * round(avg_batch_acc_charge, 2)))
            print('Test Accuracy Charge color (in progress): {}%'.format(100. * round(avg_batch_acc_chrage_color, 2)))
            print('Test Accuracy Shield color (in progress): {}%'.format(100. * round(avg_batch_acc_shield, 2)))
            
    test_epoch_loss = sum(test_epoch_loss) / len(test_epoch_loss)
    test_epoch_accuracy =  sum(test_epoch_accuracy) / len(test_epoch_accuracy)

    test_loss.append(test_epoch_loss)
    test_accuracy.append(test_epoch_accuracy)

    print("Final Test Loss = {}".format(round(test_epoch_loss, 4)))
    print("Test Accuracy Standard = {} % \n".format(test_epoch_accuracy))

    acc_test_score = sum(accuracy_test_list) / len(accuracy_test_list)
    acc_test_score_charge = sum(accuracy_test_list_charge) / len(accuracy_test_list_charge)
    acc_test_score_charge_color = sum(accuracy_test_list_charge_color) / len(accuracy_test_list_charge_color)
    acc_test_score_shield = sum(accuracy_test_list_shield) / len(accuracy_test_list_shield)
    
    print('Final Test Accuracy ALL (Overall): {}%'.format(100. * round(acc_test_score, 2)))
    print('Final Test Accuracy Charge-Mod only (Overall): {}%'.format(100. * round(acc_test_score_charge, 2)))
    print('Final Test Accuracy Charge color (Overall): {}%'.format(100. * round(acc_test_score_charge_color, 2)))
    print('Final Test Accuracy Shield color (Overall): {}%'.format(100. * round(acc_test_score_shield, 2)))


class CoAClassDataset(td.Dataset):

    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5, vocab=None, device="cpu", calc_mean=False):
        super().__init__()  # for the td.Dataset
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.df = pd.read_csv(captions_file)
        self.calc_mean = calc_mean

        # Get image and caption colum from the dataframe
        self.img_names = self.df["image"]
        self.classes = self.df["class"]

        # Get pixels colum from the dataframe
        try:
            self.psum = self.df["psum"]
        except IndexError:
            print('no pixels sum column')

        try:
            self.psum_sq = self.df["psum_sq"]
        except IndexError:
            print('no squared pixels sum column')


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Read the image and return needed information to 
        be used later by the loader

        Args:
           idx(int): index of the image we want to read in the list

        Returns:
            tensor: image tensor 
            string: image class
            float: sum of the pixels -> to calculate the mean 
            float: squared sum of the pixels -> to calculate the std
            string: image file name 
        """
        if self.calc_mean == True:
            return torch.tensor([]), torch.tensor([]), float(self.psum[idx]), float(self.psum_sq[idx]), self.img_names[idx]
        else:
            try:
                return self._get_image_tensor(idx), self._get_label_class(idx), float(self.psum[idx]), float(self.psum_sq[idx]), self.img_names[idx]
            except TypeError or IndexError:
                print(f' Error, cannot find image with index: {str(idx)}')

    def _get_image_tensor(self, idx):
        img_name = self.img_names[idx]

        img_location = os.path.join(self.root_dir, img_name)

        my_image = Path(img_location)
        if not my_image.exists():
            print(f'skipping image {img_name}, as it does not exist')

        img = Image.open(img_location).convert("RGB")
#         print(img)
        # apply the transfromation to the image
        if self.transform is not None:
#             print('self.transform is not None')
            img_t = self.transform(img)
        else:
            trans = T.ToTensor()
            img_t = trans(img)
#         print('img_t: ',img_t)

        return img_t

    def _get_label_class(self, idx):
        
        label_class = self.classes[idx]
#         print('label_class',label_class)
        
#         if self.transform is not None:
#             label_class_t = self.transform(label_class)
#         else:
#             trans = T.ToTensor()
#             label_class_t = trans(label_class)

        return label_class


if __name__ == "__main__":
    print('starting the script')
  
    parser = argparse.ArgumentParser(description='A script for training the baseline classification vgg model')
    parser.add_argument('--seed', dest='seed', type=int, help='reproducibility seed', default=1234)
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

    MISSING_TOKEN = 'None'
    
    # data_location =  '../baseline-gen-data/small/'
    data_location =  '/home/salnabulsi/coat-of-arms/data/new/'
    new_with_class_caption_file = data_location + '/new-labels-class-psumsq-2.txt'
    
    df_new = pd.read_csv(new_with_class_caption_file)
#     train, validate, test = train_validate_test_split(df_new, train_percent=.6, validate_percent=.2, seed=None)

    train_annotation_file = data_location + '/train_labels_psumsq-2.txt'
    val_annotation_file  = data_location + '/val_labels_psumsq-2.txt'
    test_annotation_file  = data_location + '/test_labels_psumsq-2.txt'

#     train.to_csv(train_annotation_file, sep=',',index=False)
#     test.to_csv(test_annotation_file, sep=',',index=False)
#     validate.to_csv(val_annotation_file, sep=',',index=False)


    # print("There are {} total images".format(len(df)))

    df1 = pd.read_csv(train_annotation_file)
    print("There are {} train images".format(len(df1)))

    df2 = pd.read_csv(val_annotation_file)
    print("There are {} val images".format(len(df2)))

    df3 = pd.read_csv(test_annotation_file)
    print("There are {} test images".format(len(df3)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 5
    NUM_WORKER = 2 #### this needs multi-core
    # NUM_WORKER = 0 #### this needs multi-core
    pin_memory=False,
    calc_mean=False
    SHUFFLE=True
    images_location = data_location + '/resized'
    learning_rate = 0.0009 #0.0005 #0.0004 #0.0001 #3e-4 #0.01 # 
    drop_prob=0.5

    train_dataset = CoAClassDataset(images_location, 
                         train_annotation_file, 
                         transform=None, 
                         device=device,
                         calc_mean=True)

    train_data_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        sampler = None,
        num_workers = NUM_WORKER,
    )
    
    height = 621
    width = 634
    
    mean = get_mean(train_dataset, train_data_loader, height, width)
    std = get_std(train_dataset, train_data_loader, mean, height, width)
    print("mean = ", mean)
    print("std = ", std)

    train_transform_list = [
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.RandomCrop(224),
    ]

    # Use RandomApply to apply the transform randomly to some of the images
    transform_with_random = T.Compose([
        T.Resize((height, width)), # mandetory                  
        random.choice(train_transform_list),
        T.ToTensor(),
        T.Normalize(mean, std), # mandetory 
        Noise(0.1, 0.05), # this should come at the end
    ])
    
    transform_for_test = T.Compose([
        T.Resize((height,width)),
        T.ToTensor(),                               
        T.Normalize(mean, std) 
    ])

    train_dataset = CoAClassDataset(images_location, 
                         train_annotation_file, 
                         transform=transform_with_random, 
                         device=device,
                         calc_mean=False)

    val_dataset = CoAClassDataset(images_location, 
                         val_annotation_file, 
                         transform=transform_with_random, 
                         device=device,
                         calc_mean=False)

    test_dataset = CoAClassDataset(images_location, 
                         test_annotation_file, 
                         transform=transform_with_random, 
                         device=device,
                         calc_mean=False)

    train_random_sampler = RandomSampler(train_dataset)
    val_random_sampler = RandomSampler(val_dataset)
    test_random_sampler = RandomSampler(test_dataset)

    # --------------------------------------------------

    # Shuffle Argument is mutually exclusive with Sampler!
    train_random_sampler = RandomSampler(train_dataset)
    val_random_sampler = RandomSampler(val_dataset)
    test_random_sampler = RandomSampler(test_dataset)

    # --------------------------------------------------

    # Shuffle Argument is mutually exclusive with Sampler!
    train_data_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        sampler = train_random_sampler,
        num_workers = NUM_WORKER,
    )

    val_data_loader = DataLoader(
        dataset = val_dataset,
        batch_size = BATCH_SIZE,
        sampler = val_random_sampler,
        num_workers = NUM_WORKER,
    )

    test_data_loader = DataLoader(
        dataset = test_dataset,
        batch_size = BATCH_SIZE,
        sampler = test_random_sampler,
        num_workers = NUM_WORKER,
    )    
    # --------------------------------------------------
    ### Define model
    model = models.vgg16(pretrained = True)

    ### Modifying last few layers and no of classes
    # NOTE: cross_entropy loss takes unnormalized op (logits), then function itself applies softmax and calculates loss, so no need to include softmax here
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096, bias = True),
        nn.ReLU(inplace = True),
        nn.Dropout(drop_prob),
        nn.Linear(4096, 2048, bias = True),
        nn.ReLU(inplace = True),
        nn.Dropout(drop_prob),
        nn.Linear(2048, 200)
    )

    # --------------------------------------------------

    torch.cuda.empty_cache()

    model.to(device)

    ### Training Details

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.75)
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    train_accuracy = []

    val_loss = []
    val_accuracy = []

    epochs = 100
    CUDA_LAUNCH_BLOCKING=1

    # --------------------------------------------------

    model, train_epoch_loss, train_epoch_accuracy, val_epoch_loss, val_epoch_accuracy = train_classification_model(model,optimizer, criterion, lr_scheduler, epochs, train_data_loader, val_data_loader, device)


    # --------------------------------------------------
    
    from datetime import datetime

    # save the latest model
    now = datetime.now() # current date and time
    timestr = now.strftime("%m-%d-%Y-%H:%M:%S")
    model_full_path = f"{data_location}/classification-model-trained-on-only-real-{timestr}.pth"
    
    model.cpu()
    model_state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_epoch_loss,
        'accuracy': train_epoch_accuracy
    }

    torch.save(model_state, model_full_path)
    print('model has been saved to: ', model_full_path)
    
    print('running test directly afterwards')
    test_classification_model(model, test_data_loader)
