import os
import torch
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from src.baseline.vocabulary import Vocabulary

class CoADataset(Dataset):
 
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5, vocab=None, device="cpu"):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.device = device
        #Get image and caption colum from the dataframe
        self.img_names = self.df["image"]
        self.captions = self.df["caption"]
        
        #Initialize vocabulary and build vocab
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocab(self.captions.tolist())
        else: 
            self.vocab = vocab
        
        self.images, self.caption_vecs = self.load_images()

    def load_images(self):
        n = len(self.img_names)
        images = []
        caption_vecs = []

        for idx in range(n):
            caption = self.captions[idx]
            img_name = self.img_names[idx]
            
            img_location = os.path.join(self.root_dir, img_name)
            img = Image.open(img_location).convert("RGB")
            
            #apply the transfromation to the image
            if self.transform is not None:
                img = self.transform(img)
            else:
                trans = T.ToTensor()
                img = trans(img)
            
            # numericalize the caption text
            caption_vec = []
            caption_vec += [self.vocab.stoi["<SOS>"]]
            caption_vec += self.vocab.numericalize(caption)
            caption_vec += [self.vocab.stoi["<EOS>"]]

            images.append(img.to(self.device))
            caption_vecs.append(torch.tensor(caption_vec).to(self.device))     
            
            return images, caption_vecs

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
#         try:
        return self.images[idx], self.caption_vecs[idx]
#         except TypeError or IndexError:
#             print(str(idx))
