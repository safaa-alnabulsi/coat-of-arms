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
            self.psum = self.df["psum"]
        except IndexError:
            print('no pixels sum column')
            
        try:
            self.psum_sq = self.df["psum_sq"]
        except IndexError:
            print('no squared pixels sum column')

        #Initialize vocabulary and build vocab
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocab(self.captions.tolist())
        else: 
            self.vocab = vocab
            
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Read the image and return needed information to 
        be used later by the loader

        Args:
           idx(int): index of the image we want to read in the list

        Returns:
            tensor: image tensor 
            string: image caption
            float: sum of the pixels -> to calculate the mean 
            float: squared sum of the pixels -> to calculate the std
            string: image file name 
        """
        if self.calc_mean == True:
            return torch.tensor([]), torch.tensor([]),float(self.psum[idx]), float(self.psum_sq[idx]), self.img_names[idx]
        else:
            try:
                return self._get_image_tensor(idx), self._get_caption_vec(idx), float(self.psum[idx]),float(self.psum_sq[idx]), self.img_names[idx]
            except TypeError or IndexError:
                print(f' Error, cannot find image with index: {str(idx)}')


    def _get_image_tensor(self, idx):
        img_name = self.img_names[idx]
        
        img_location = os.path.join(self.root_dir, img_name)
        
        my_image = Path(img_location)
        if not my_image.exists():
            print(f'skipping image {img_name}, as it does not exist')
        
        img = Image.open(img_location).convert("RGB")
        
        # # resize the image t0 100x100 to improve the iteration time
        # crops_size = 100,100
        # img.thumbnail(crops_size, Image.ANTIALIAS)

        # apply the transfromation to the image
        if self.transform is not None:
            img_t = self.transform(img)
        else:
            trans = T.ToTensor()
            img_t = trans(img)
                     
        return img_t
           
                
    def _get_caption_vec(self, idx):

        caption = self.captions[idx]

        # numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return torch.tensor(caption_vec)   
