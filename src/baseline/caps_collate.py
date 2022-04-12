import torch
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False,calc_mean=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        self.calc_mean = calc_mean
    
    def __call__(self,batch):
        imgs,targets,pixels = [],[],[]
        if self.calc_mean == True:
            for item in batch:
                pixels.append(item[2])
            imgs = torch.tensor([])
            targets = torch.tensor([])      
        else:    
            for item in batch:
                imgs.append(item[0].unsqueeze(0))
                targets.append(item[1])
                pixels.append(item[2])
            
            imgs = torch.cat(imgs,dim=0)
            targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        
        pixels = torch.tensor(pixels)
            
        return imgs,targets,pixels