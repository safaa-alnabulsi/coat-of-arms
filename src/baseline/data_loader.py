import torch
import torchvision
import torchdatasets as td

from torch.utils.data import DataLoader,Dataset
from src.baseline.vocabulary import Vocabulary
from src.baseline.coa_dataset import CoADataset
from src.baseline.caps_collate import CapsCollate
from src.utils import print_time

def get_loader(root_folder, annotation_file, transform, 
               batch_size=32, num_workers=2, shuffle=True, 
               pin_memory=True, vocab=None, device='cpu', calc_mean=False):
    print('before CoADataset init')
    dataset = CoADataset(root_folder, 
                         annotation_file, 
                         transform=transform, 
                         vocab=vocab,
                         device=device,
                         calc_mean=calc_mean)
#     .map(torchvision.transforms.ToTensor()).cache(td.modifiers.UpToIndex(500, td.cachers.Memory())).cache(td.modifiers.FromIndex(500, td.cachers.Pickle("./cache")))
#     .cache()
# #                         # First 1000 samples in memory
#                         .cache(td.modifiers.UpToIndex(500, td.cachers.Memory()))
# #                         # Sample from 1000 to the end saved with Pickle on disk
#                         .cache(td.modifiers.FromIndex(500, td.cachers.Pickle("./cache")))
# #                         # You can define your own cachers, modifiers, see docs

    print('after CoADataset init')
    pad_idx = dataset.vocab.stoi["<PAD>"]
  
    print('before DataLoader init')
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CapsCollate(pad_idx=pad_idx,calc_mean=calc_mean)
    )
    print('after DataLoader init')

    return loader, dataset

def get_loaders(root_folder, train_annotation_file, val_annotation_file, test_annotation_file, 
                transform, batch_size=32, num_workers=2, shuffle=True, pin_memory=True, vocab=None, device='cpu'):
    print('initing train loader')
    train_loader, train_dataset = get_loader(root_folder, train_annotation_file, transform, 
                                             batch_size, num_workers,shuffle, pin_memory, vocab, device)
    print('-------------')                                             
    print('initing val loader')
    val_loader, val_dataset = get_loader(root_folder, val_annotation_file, transform, 
                                         batch_size, num_workers,shuffle, pin_memory, vocab, device)
    print('-------------')                                             
    print('initing test loader')
    test_loader, test_dataset = get_loader(root_folder, test_annotation_file, transform, 
                                           batch_size, num_workers,shuffle, pin_memory, vocab, device)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def get_mean(train_dataset, train_loader, img_h=500, img_w=500):
    num_of_pixels = len(train_dataset) * img_h * img_w

    total_sum = 0

    # the batch[0] contains the images, batch[1] contains labels and 2 contains pixels values.
    for _, batch in enumerate(iter(train_loader)):
        total_sum += batch[2].sum()
    
    mean = total_sum / num_of_pixels
    
    return mean

# Following new way in calculating std
# https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html?fbclid=IwAR2QnsqPGzyOkKw6yYtMMB1mmsNC8dIYt_6HdoKoOs9vFSciULBfDGIQ7Kw#:~:text=mean%3A%20simply%20divide%20the%20sum,%2F%20count%20%2D%20total_mean%20**%202

def get_std(train_dataset, train_loader, mean, img_h=500, img_w=500):
    count_of_pixels = len(train_dataset) * img_h * img_w
    
    for _, batch in enumerate(iter(train_loader)):
        var += (batch[3] / count_of_pixels) - (mean ** 2)
  
    std = torch.sqrt(var)
    
    return std
    