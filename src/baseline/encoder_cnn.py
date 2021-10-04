import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        

    def forward(self, images):
        features = self.resnet(images)                                    #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        
        # features1 torch.Size([10, 2048, 7, 7])
        # features2 torch.Size([10, 7, 7, 2048]) # the LSTM hidden state is batch_size,decoder_dim feature extracted by the Conv net is batch_size,2048,7,7
        # features3 torch.Size([10, 49, 2048]) # batch_size=10,num_features=49,dim=2048
        return features
