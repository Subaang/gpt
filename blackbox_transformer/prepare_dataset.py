import torch

class Data:
    def __init__(self, path, block_size):
        self.block_size = block_size
        self.text = ""
        with open(path) as f:
            self.text = f.read()
        
        self.vocab = list(set(self.text))
        self.vocab_size = len(self.vocab)
        self.encode = {val:i for i,val in enumerate(self.vocab)}
        self.decode = {i:val for i,val in enumerate(self.vocab)}

        self.tokenized_text = torch.tensor([self.encode[c] for c in self.text],dtype=torch.long)

        self.train_data = self.tokenized_text[:int(0.9*len(self.text))]
        self.val_data = self.tokenized_text[int(0.9*len(self.text)):]

        
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data

        random_starting_indices = torch.randint(len(data) - self.block_size, (self.block_size,))

        x = torch.stack([data[i:i+self.block_size] for i in random_starting_indices])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in random_starting_indices])
        return x,y
        

if __name__ == "__main__":
    d = Data()