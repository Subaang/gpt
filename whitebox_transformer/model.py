import torch.nn as nn
import torch.nn.functional as F
import torch

class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size, max_seq_length):
        super().__init__()
        self.input_dim = d_model

        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        
        self.head_size = head_size
        
        # Mask buffer
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_length, max_seq_length))) #(max_seq_length, max_seq_length)
        
    def forward(self, x):
            B,T,C = x.shape

            # Q @ K_T is "read only", thus we can apply mask after this step
            
            Q = self.query(x) # (B, T, head_size)
            K = self.key(x)   # (B, T, head_size)
            V = self.value(x) # (B, T, head_size)
            
            K_T = K.transpose(-2, -1) # (B, head_size, T)
            
            scores = Q @ K_T # (B,T,head_size) @ (B,head_size,T) -> (B, T, T)
            scores = scores / (self.head_size ** 0.5) 
 
            scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            
            weights = F.softmax(scores, dim=-1) # (B, T, T)     
            out = weights @ V # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
            
            return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_length):
        super().__init__()
        self.head_size = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.head_size, max_seq_length) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(d_model, d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Pre-LN
        x_norm = self.ln1(x)
        
        head_outputs = [head(x_norm) for head in self.heads]
        mha_out = torch.cat(head_outputs, dim=-1)
        proj_out = self.proj(mha_out)
        
        # Pure residual addition (No normalization here)
        x = x + proj_out

        x_norm2 = self.ln2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out

        return x
        

class LanguageModel(nn.Module):
    def __init__(self,vocab_size, d_model, max_seq_length, n_heads, n_layers): 
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_embedding = nn.Embedding(num_embeddings=max_seq_length,embedding_dim=d_model)

        self.layers = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, max_seq_length) for _ in range(n_layers)]
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.max_seq_length = max_seq_length


    def forward(self,input_encodings,targets_encodings=None):
        B,T = input_encodings.shape

        positions = torch.arange(T, device=input_encodings.device)

        input_embeddings = self.embedding(input_encodings) + self.pos_embedding(positions)
        transform = self.layers(input_embeddings)
        out = self.lm_head(transform) # (B,T,vocab_size)

        loss = None
        if targets_encodings is not None:
            out_for_loss = out.permute(0,2,1) # (B,vocab_size,T)
            loss = F.cross_entropy(out_for_loss,targets_encodings)
        
        return loss, out 
    
    def generate(self,input_encoding, max_output_len):
        for _ in range(max_output_len):
            input_cropped = input_encoding[:, -self.max_seq_length:]
            _,logits = self(input_cropped)
            logits = logits[:,-1,:] # Last char of sequence
            predictions_probs = F.softmax(logits,dim=-1)
            prediction = torch.multinomial(predictions_probs, num_samples=1)
            input_encoding = torch.cat((input_encoding,prediction),dim=1)

        return input_encoding
