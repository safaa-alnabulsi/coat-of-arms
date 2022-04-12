import os
import torch
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from src.baseline.vocabulary import Vocabulary
import torchdatasets as td
from src.utils import print_time
from pathlib import Path


class CoADataset(td.Dataset):
 
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5, vocab=None, device="cpu", calc_mean=False):
        super().__init__() # for the td.Dataset
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.df = pd.read_csv(captions_file)
        self.calc_mean = calc_mean

        # Get image and caption colum from the dataframe
        self.img_names = self.df["image"]
        self.captions = self.df["caption"]

        # Get pixels colum from the dataframe       
        try:
            self.pixels = self.df["pixels"]
        except IndexError:
            print('no pixels columns')

        #Initialize vocabulary and build vocab
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocab(self.captions.tolist())
        else: 
            self.vocab = vocab
            
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.calc_mean == True:
           return torch.tensor([]),torch.tensor([]),float(self.pixels[idx])
        else:
            try:
                return self._get_image_tensor(idx), self._get_caption_vec(idx), float(self.pixels[idx])
            except TypeError or IndexError:
                print(f' Error, cannot find image with index: {str(idx)}')

    def _get_image_tensor(self, idx):
        img_name = self.img_names[idx]
        
        img_location = os.path.join(self.root_dir, img_name)
        
        my_image = Path(img_location)
        if not my_image.exists():
            print(f'skipping image {img_name}, as it does not exist')

        img = Image.open(img_location).convert("RGB")
        
        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)
        else:
            trans = T.ToTensor()
            img = trans(img)
        return img
                
    def _get_caption_vec(self, idx):

        caption = self.captions[idx]

        # numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return torch.tensor(caption_vec)   
