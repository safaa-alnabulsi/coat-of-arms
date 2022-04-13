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
        imgs,targets,psum, psum_sq= [],[],[],[]
        if self.calc_mean == True:
            for item in batch:
                psum.append(item[2])
                psum_sq.append(item[3])

            imgs = torch.tensor([])
            targets = torch.tensor([])      
        else:    
            for item in batch:
                imgs.append(item[0].unsqueeze(0))
                targets.append(item[1])
                psum.append(item[2])
                psum_sq.append(item[3])
            
            imgs = torch.cat(imgs,dim=0)
            targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        
        psum = torch.tensor(psum)
        psum_sq = torch.tensor(psum)
            
        return imgs,targets,psum,psum_sq