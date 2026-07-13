import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramModel(nn.Module):
    def __init__(self,n_embeddings, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=vocab_size)

    def forward(self,input_encodings,targets_encodings=None):
        input_embeddings = self.embedding(input_encodings) # (B,T,C) , B = block_size, T = length of sequence, C = embedding_dim

        loss = None
        if targets_encodings is not None:
            input_embeddings_for_loss = input_embeddings.permute(0,2,1) # (B,C,T)
            loss = F.cross_entropy(input_embeddings_for_loss,targets_encodings)
        
        return loss, input_embeddings 
    
    def generate(self,input_encoding, max_output_len):
        for _ in range(max_output_len):
            _,logits = self(input_encoding)
            logits = logits[:,-1,:] # Last char of sequence
            predictions_probs = F.softmax(logits,dim=-1)
            prediction = torch.multinomial(predictions_probs, num_samples=1)
            input_encoding = torch.cat((input_encoding,prediction),dim=1)

        return input_encoding
    
