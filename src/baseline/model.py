import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(EncoderCNN, self).__init__()
        # create a pretrained ResNet-xx model
        # resnet = models.resnet50(pretrained=pretrained) # encoder_dim -> 2048
        # resnet = models.resnet34(pretrained=pretrained) # encoder_dim -> 512
        resnet = models.resnet18(pretrained=pretrained)   # encoder_dim -> 512
        
        # freeze the layers by stopping them from accumulating gradients by using requires_grad()
        # We need to do this for every parameter in the network.
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        #  Get all layers of the network except the last two items
        modules = list(resnet.children())[:-2] 
        
        # assign back to the network without the last two layers 
        self.resnet = nn.Sequential(*modules)
        

    def forward(self, images):
        features = self.resnet(images)                                    #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        
        # features1 torch.Size([10, 2048, 7, 7])
        # features2 torch.Size([10, 7, 7, 2048]) # the LSTM hidden state is batch_size,decoder_dim feature extracted by the Conv net is batch_size,2048,7,7
        # features3 torch.Size([10, 49, 2048]) # batch_size=10,num_features=49,dim=2048
        return features

#Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, encoder_dim,decoder_dim,attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)
        
        self.A = nn.Linear(attention_dim,1)
        
    def forward(self, features, hidden_state):
        u_hs = self.U(features)     #(batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state) #(batch_size,attention_dim)
        
        # Calculating Alignment Scores
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        
        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)
        
         # Softmaxing alignment scores to get Attention weights   
        alpha = F.softmax(attention_scores,dim=1)          #(batch_size,num_layers)
        
        # Multiplying the Attention weights with encoder outputs to get the context vector
        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)   #(batch_size,num_layers)
        
        return alpha,attention_weights


#Attention Decoder
class DecoderRNN(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        
        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        # double check with david if we need this embedding, the caption is already tokenized and comes as a vector of number
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        self.fcn = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def forward(self, features, captions):
        
        # vectorize the caption by embedding input words
        embeds = self.embedding(captions)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = len(captions[0])-1 #Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)
        
#         print('seq_length: ',seq_length) # seq_length:  9
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(self.device)

        for s in range(seq_length):
            alpha,context = self.attention(features, h)
            # print(embeds.shape) 
            # print(context.shape)
            
            # print(f'seq_length {seq_length}')
            # pad_sequence batch_first True
            # embeds torch.Size([9, 10, 300])
            # context torch.Size([10, 512])

            # pad_sequence batch_first False
            # embeds torch.Size([10, 9, 300])
            # context torch.Size([10, 512])
            
            # https://pytorch.org/docs/stable/generated/torch.cat.html
            lstm_input = torch.cat((embeds[:, s], context), dim=1) ##### <=== breaking here
            h, c = self.lstm_cell(lstm_input, (h, c))
                    
            output = self.fcn(self.drop(h))
            
            preds[:,s] = output
            alphas[:,s] = alpha  
        
        return preds, alphas
    
    def generate_caption(self,features,max_len=20,vocab=None):
        # Inference part
        # Given the image features generate the captions
        
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        alphas = []
        
        #starting input
        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to(self.device)
        embeds = self.embedding(word)
        
        captions = []
        
        for i in range(max_len):
            alpha,context = self.attention(features, h)
            
            
            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)
        
            
            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
            captions.append(predicted_word_idx.item())
            
            #end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            
            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
        
        #covert the vocab idx to words and return sentence
        return [vocab.itos[idx] for idx in captions],alphas
    
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
    
class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3, pretrained=True):
        super().__init__()
        self.encoder = EncoderCNN(pretrained)
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            drop_prob = drop_prob
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        # print(outputs)
        return outputs
