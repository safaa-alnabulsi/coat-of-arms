#!/usr/bin/python

#imports 
import os
# import torch
# import spacy
# import pandas as pd
# import numpy as np
# import torchvision.transforms as T
# import matplotlib.pyplot as plt
# import torch
# import nltk
# import numpy as np
# import torch.nn as nn
# import torch.onnx as onnx
# import torch.optim as optim
# import torchvision.models as models
# from torch.utils.data import DataLoader,Dataset
# from PIL import Image
# from datetime import datetime
# from tqdm import tqdm
# from time import sleep
# from torch.utils.data import DataLoader,Dataset

# from src.baseline.model import EncoderCNN, Attention, DecoderRNN, EncoderDecoder
# from src.baseline.vocabulary import Vocabulary
# from src.baseline.coa_dataset import CoADataset
# from src.baseline.caps_collate import CapsCollate
# from src.baseline.data_loader import get_loader, get_loaders, get_mean_std
# from src.accuracy import Accuracy

def print_time(text):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("{} @ Time = {}".format(text, current_time))

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

def get_new_model():
    model = EncoderDecoder(embed_size, len(train_dataset.vocab), attention_dim, encoder_dim, decoder_dim, drop_prob=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])

    return model, optimizer, criterion


# Function to test the model with the val dataset and print the accuracy for the test images
def validate(model):
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

#             print('correct_caption: ', correct_caption_s)
#             print('predicted_caption: ', predicted_caption)
            
#             show_image(img[0], title=predicted_caption)
                    
            # ------------------------------------------
            # calc metrics
            accuracy_list.append(Accuracy(predicted_caption,correct_caption_s).get())
            
            bleu = nltk.translate.bleu_score.sentence_bleu([correct_caption], caps, weights=(0.5, 0.5))
            bleu_score += bleu

            # ------------------------------------------
            # calc losses and take the average 
            image, captions = img.to(device), correct_cap.to(device)
            outputs, _ = model(image, captions.T)
            targets = captions.T[:,1:] 
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            val_losses.append(loss)
            
            # ------------------------------------------
           
    # compute the accuracy over all test images
    acc_score = (100 * sum(accuracy_list) / len(accuracy_list))
    avg_loss = sum(val_losses) / len(val_losses)

    return avg_loss, bleu_score, acc_score

#helper function to save the model
def save_model(model, optimizer, epoch, loss, accuracy, model_full_path):
    model.cpu()
    model_state = {
        'epoch': epoch,
        'embed_size': embed_size,
        'vocab_size': len(train_dataset.vocab),
        'attention_dim': attention_dim,
        'encoder_dim': encoder_dim,
        'decoder_dim': decoder_dim,
        'state_dict': model.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy':accuracy
    }
    
    torch.save(model_state, model_full_path)

if __name__ == "__main__":
    print('starting the script')
#     data_location = '/home/space/datasets/COA/generated-data-api'
#     data_location = '/home/space/datasets/COA/generated-data-api-small'

    
#     caption_file = data_location + '/captions.txt'
#     root_folder_images = data_location + '/images'
#     df = pd.read_csv(caption_file)

#     train, validate, test = train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None)


#     train_annotation_file = data_location + '/train_captions.txt'
#     val_annotation_file  = data_location + '/val_captions.txt'
#     test_annotation_file  = data_location + '/test_captions.txt'

#     train.to_csv(train_annotation_file, sep=',',index=False)
#     test.to_csv(test_annotation_file, sep=',',index=False)
#     validate.to_csv(val_annotation_file, sep=',',index=False)


#     print("There are {} total images".format(len(df)))

#     caption_file = data_location + '/train_captions.txt'
#     df1 = pd.read_csv(caption_file)
#     print("There are {} train images".format(len(df1)))

#     caption_file = data_location + '/val_captions.txt'
#     df2 = pd.read_csv(caption_file)
#     print("There are {} val images".format(len(df2)))

#     caption_file = data_location + '/test_captions.txt'
#     df3 = pd.read_csv(caption_file)
#     print("There are {} test images".format(len(df3)))
    
#     # -------------------------------------------------------------------------------------------------------
    
#     #setting the constants
#     BATCH_SIZE = 10
#     NUM_WORKER = 10 #### this needs multi-core
#     freq_threshold = 5
    
#     # 30 minutes to create those, as it's baseline, i ran it several times and it's the same
#     vocab = Vocabulary(freq_threshold)
#     vocab.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'g': 4, 'v': 5, 'b': 6, 'cross': 7, 'lion': 8, 'passt': 9, 's': 10, 'a': 11, 'eagle': 12, 'o': 13, 'doubleheaded': 14, "'s": 15, 'head': 16, 'patonce': 17, 'moline': 18, 'guard': 19, 'rampant': 20}
#     vocab.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>', 4: 'g', 5: 'v', 6: 'b', 7: 'cross', 8: 'lion', 9: 'passt', 10: 's', 11: 'a', 12: 'eagle', 13: 'o', 14: 'doubleheaded', 15: "'s", 16: 'head', 17: 'patonce', 18: 'moline', 19: 'guard', 20: 'rampant'}
    
#     # -------------------------------------------------------------------------------------------------------
    
#     #Initiate the Dataset and Dataloader
#     train_loader, train_dataset = get_loader(
#         root_folder=root_folder_images,
#         annotation_file=train_annotation_file,
#         transform=None,  # <=======================
#         num_workers=NUM_WORKER,
#         vocab=vocab,
#         batch_size=50
#     )
#     mean, std = get_mean_std(train_dataset, train_loader, 500 , 500)
#     print('mean, std:', mean, std)
#     #defining the transform to be applied

#     transform = T.Compose([
#         T.Resize(226),                     
#         T.RandomCrop(224),                 
#         T.ToTensor(),                               
#         T.Normalize(mean, std) 
#     ])

#     print_time('writing the dataloader')

#     train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_loader(
#         root_folder=root_folder_images,
#         train_annotation_file=train_annotation_file,
#         val_annotation_file=val_annotation_file,
#         test_annotation_file=test_annotation_file,
#         transform=transform,
#         num_workers=NUM_WORKER,
#         vocab=vocab,
#         batch_size=BATCH_SIZE
#     )

#     print_time('finished writing the dataloader')
#     # -------------------------------------------------------------------------------------------------------

#     #Hyperparams
#     embed_size=300
#     vocab_size = len(train_dataset.vocab)
#     attention_dim=256
#     encoder_dim=2048
#     decoder_dim=512
#     learning_rate = 3e-4
    
#     # -------------------------------------------------------------------------------------------------------

#     #initialize new model, loss etc
#     model, optimizer, criterion = get_new_model()    

#     losses = list()
#     losses_batch = list()
#     val_losses = list()
#     accuracy_list = list()

#     model_full_path = '/home/space/datasets/COA/models/baseline/attention_model_with_acc.pth'
#     num_epochs = 5
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for epoch in range(1, num_epochs + 1): 
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
#     #             print(targets)
#                 loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
#     #             print(outputs.view(-1, vocab_size))

#                 # Backward pass. 
#                 loss.backward()

#                 # Update the parameters in the optimizer.
#                 optimizer.step()

#                 tepoch.set_postfix(loss=loss.item())
#                 sleep(0.1)

#                 avg_val_loss, bleu_score, accuracy = validate(model)
#                 model.train()
#     #             tepoch.set_postfix(accuracy=accuracy)
#     #             tepoch.set_postfix(vallidation_loss=avg_val_loss)

#                 losses_batch.append(loss) # in here 17 batches * 5 epochs = 85 , you can get the average
#                 val_losses.append(avg_val_loss)
#                 accuracy_list.append(accuracy)

#             avg_batch_loss = sum(losses_batch) / len(losses_batch)
#             losses.append(avg_batch_loss)
            
#         epoch_accuracy = sum(accuracy_list)/len(accuracy_list)
#         epoch_loss = sum(losses)/len(losses)
#         save_model(model, optimizer, epoch, epoch_loss, epoch_accuracy, model_full_path)


#     print('Bleu Score: ',bleu_score/8091)
#     print('Final accuracy: ', sum(accuracy_list)/len(accuracy_list))
