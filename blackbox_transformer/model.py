import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlackbox(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, block_size):
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.lin = nn.Linear(d_model, vocab_size)

    def forward(self, input_encodings, target_encodings=None):
        B, T = input_encodings.shape

        
        tok_emb = self.token_embedding(input_encodings) # (B, T) -> (B, T, d_model)
        
        positions = torch.arange(0, T, dtype=torch.long, device=input_encodings.device)
        pos_emb = self.pos_embedding(positions) 

        x = tok_emb + pos_emb 
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_encodings.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        
        logits = self.lin(x) # (B, T, vocab_size)
        loss = None

        if target_encodings is not None:
            logits_for_loss = logits.permute(0, 2, 1)
            loss = F.cross_entropy(logits_for_loss, target_encodings)

        return loss, logits

    def generate(self, input_encoding, max_output_len):
        for _ in range(max_output_len):
            input_cond = input_encoding[:, -self.block_size:] 
            
            _, logits = self(input_cond)
            logits = logits[:, -1, :] 
            predictions_probs = F.softmax(logits, dim=-1)
            prediction = torch.multinomial(predictions_probs, num_samples=1)
            input_encoding = torch.cat((input_encoding, prediction), dim=1)

        return input_encoding