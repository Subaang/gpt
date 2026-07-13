from prepare_dataset import Data
from model import LanguageModel
import torch
import torch.optim as optim

def train(model, n_steps, optimizer, device):
    model.train()
    
    for step in range(n_steps):
        x, y = data.get_batch('train')
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()       
        loss, _ = model(x, y)   
        loss.backward()         
        optimizer.step()            
        
        if step % 100 == 0:
            print(f"Step {step} | Train Loss = {loss.item():.4f}")

def evaluate(model,device):
    model.eval()
    x,y = data.get_batch('eval')
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        loss, _ = model(x, y)
    print("Validation Loss = ",loss.item())


if __name__ == "__main__":
    data = Data("../tinyshakesphere.txt", 128)
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    vocab_size = data.vocab_size
    encode = data.encode
    decode = data.decode
    block_size = data.block_size
    model = LanguageModel(
        vocab_size=vocab_size, 
        d_model=512, 
        max_seq_length=block_size, 
        n_heads=8, 
        n_layers=8
    )
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    train(model, 400, optimizer,device)
    evaluate(model,device)

    inp_text = "Hello"
    inp_text_tokenized = torch.tensor([encode[c] for c in inp_text], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        gen = model.generate(inp_text_tokenized, 100)
    print(''.join([decode[int(i)] for i in gen[0]]))

    torch.save(model.state_dict(), 'blackbox_transformer_model.pt')
    print("Model successfully saved!")


