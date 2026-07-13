from prepare_dataset import Data
from model import BigramModel
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
        loss,_ = model(x,y)
    print("Validation Loss = ",loss.item())


if __name__ == "__main__":
    data = Data("../tinyshakesphere.txt", 128)
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    vocab_size = data.vocab_size
    encode = data.encode
    decode = data.decode
    model = BigramModel(vocab_size, vocab_size) # 1 row for each possible char, and for softmax, we need the vocab as output
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    train(model, 4000, optimizer,device)
    evaluate(model,device)
    
    start_context = torch.zeros((1, 1), dtype=torch.long).to(device)
    with torch.no_grad():
        gen = model.generate(start_context, 100)
    print(''.join([decode[int(i)] for i in gen[0]]))


