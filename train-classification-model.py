#!/usr/bin/python

#imports 

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

def calc_accuracy(true,pred):
    pred = F.softmax(pred, dim = 1)
    true = torch.zeros(pred.shape[0], pred.shape[1]).scatter_(1, true.unsqueeze(1), 1.)
    acc = (true.argmax(-1) == pred.argmax(-1)).float().detach().numpy()
    acc = float((100 * acc.sum()) / len(acc))
    return round(acc, 4)

### Training Code
def train_classification_model(model, epochs, train_data_loader, val_data_loader, device):
    from tqdm import tqdm

    for epoch in range(epochs):
        start = time.time()

        #Epoch Loss & Accuracy
        train_epoch_loss = []
        train_epoch_accuracy = []
        _iter = 1

        #Val Loss & Accuracy
        val_epoch_loss = []
        val_epoch_accuracy = []

        # Training
        with tqdm(train_data_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for images, labels,_,_,_ in tepoch: 
                images = images.to(device)
                labels = labels.to(device)

                #Reset Grads
                optimizer.zero_grad()

                #Forward ->
                preds = model(images)

                #Calculate Accuracy
                acc = calc_accuracy(labels.cpu(), preds.cpu())

                #Calculate Loss & Backward, Update Weights (Step)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

                #Append loss & acc
                loss_value = loss.item()
                train_epoch_loss.append(loss_value)
                train_epoch_accuracy.append(acc)

                tepoch.set_postfix({'train_epoch_loss': loss_value})
    #             tepoch.set_postfix({'train_epoch_accuracy',acc})

                if _iter % 500 == 0:
                    print("> Iteration {} < ".format(_iter))
                    print("Iter Loss = {}".format(round(loss_value, 4)))
                    print("Iter Accuracy = {} % \n".format(acc))

    #             _iter += 1

        #Validation
        with tqdm(val_data_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for images, labels,_,_,_ in tepoch:
                images = images.to(device)
                labels = labels.to(device)

                #Forward ->
                preds = model(images)

                #Calculate Accuracy
                acc = calc_accuracy(labels.cpu(), preds.cpu())

                #Calculate Loss
                loss = criterion(preds, labels)

                #Append loss & acc
                loss_value = loss.item()
                val_epoch_loss.append(loss_value)
                val_epoch_accuracy.append(acc)
                tepoch.set_postfix({'val_epoch_loss': loss_value})
    #             tepoch.set_postfix({'val_epoch_accuracy',acc})


        train_epoch_loss = np.mean(train_epoch_loss)
        train_epoch_accuracy = np.mean(train_epoch_accuracy)

        val_epoch_loss = np.mean(val_epoch_loss)
        val_epoch_accuracy = np.mean(val_epoch_accuracy)

        end = time.time()

        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)

        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

        #Print Epoch Statistics
        print("** Epoch {} ** - Epoch Time {}".format(epoch, int(end-start)))
        print("Train Loss = {}".format(round(train_epoch_loss, 4)))
        print("Train Accuracy = {} % \n".format(train_epoch_accuracy))
        print("Val Loss = {}".format(round(val_epoch_loss, 4)))
        print("Val Accuracy = {} % \n".format(val_epoch_accuracy))
    
    return model, train_epoch_loss, train_epoch_accuracy, val_epoch_loss, val_epoch_accuracy

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
#         if self.transform is not None:
#             label_class_t = self.transform(label_class)
#         else:
#             trans = T.ToTensor()
#             label_class_t = trans(label_class)

        return label_class


if __name__ == "__main__":
    print('starting the script')

    MISSING_TOKEN = 'None'
    
    # data_location =  '../baseline-gen-data/small/'
    data_location =  '/home/space/datasets/COA/generated-data-api-single-and-plural/'
    new_with_class_caption_file = data_location + '/new-labels-class-psumsq.txt'

    train, validate, test = train_validate_test_split(df_new, train_percent=.6, validate_percent=.2, seed=None)

    train_annotation_file = data_location + '/train_labels_psumsq.txt'
    val_annotation_file  = data_location + '/val_labels_psumsq.txt'
    test_annotation_file  = data_location + '/test_labels_psumsq.txt'

    train.to_csv(train_annotation_file, sep=',',index=False)
    test.to_csv(test_annotation_file, sep=',',index=False)
    validate.to_csv(val_annotation_file, sep=',',index=False)


    # print("There are {} total images".format(len(df)))

    df1 = pd.read_csv(train_annotation_file)
    print("There are {} train images".format(len(df1)))

    df2 = pd.read_csv(val_annotation_file)
    print("There are {} val images".format(len(df2)))

    df3 = pd.read_csv(test_annotation_file)
    print("There are {} test images".format(len(df3)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 56
    NUM_WORKER = 2 #### this needs multi-core
    # NUM_WORKER = 0 #### this needs multi-core
    pin_memory=False,
    calc_mean=False
    SHUFFLE=True
    images_location = data_location + '/res_images'

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

    mean = get_mean(train_dataset, train_data_loader, 500 , 500)
    std = get_std(train_dataset, train_data_loader, mean, 500 , 500)
    print("mean = ", mean)
    print("std = ", std)

    transform = T.Compose([
        T.Resize(226),                     
        T.RandomCrop(224),                 
        T.ToTensor(),                               
        T.Normalize(mean, std)
    ])

    train_dataset = CoAClassDataset(images_location, 
                         train_annotation_file, 
                         transform=transform, 
                         device=device,
                         calc_mean=False)

    val_dataset = CoAClassDataset(images_location, 
                         val_annotation_file, 
                         transform=transform, 
                         device=device,
                         calc_mean=False)

    test_dataset = CoAClassDataset(images_location, 
                         test_annotation_file, 
                         transform=transform, 
                         device=device,
                         calc_mean=False)

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
        nn.Dropout(0.4),
        nn.Linear(4096, 2048, bias = True),
        nn.ReLU(inplace = True),
        nn.Dropout(0.4),
        nn.Linear(2048, 200)
    )

    # --------------------------------------------------

    torch.cuda.empty_cache()

    model.to(device)

    ### Training Details

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.75)
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    train_accuracy = []

    val_loss = []
    val_accuracy = []

    epochs = 25
    CUDA_LAUNCH_BLOCKING=1

    # --------------------------------------------------

    model, train_epoch_loss, train_epoch_accuracy, val_epoch_loss, val_epoch_accuracy = train_classification_model(model, epochs, train_data_loader, val_data_loader, device)


    # --------------------------------------------------

    
    from datetime import datetime

    # save the latest model
    now = datetime.now() # current date and time
    timestr = now.strftime("%m-%d-%Y-%H:%M:%S")
    model_full_path = f"{data_location}/classification-model-{timestr}.pth"

    model.cpu()
    model_state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_epoch_loss,
        'accuracy': train_epoch_accuracy
    }

    torch.save(model_state, model_full_path)
