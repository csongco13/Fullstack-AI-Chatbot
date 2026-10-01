import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?','!','.',',',"'s"]
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))  # Associated number for each tag
print(f"{len(xy)} patterns | {len(tags)} tags | {len(all_words)} unique stemmed words")

X_train=[]  # Here, we put the bag of words
y_train=[]  
for (pattern_sentence, tag) in xy:   # Unpack the tuple
    X_train.append(bag_of_words(pattern_sentence,all_words))    # Here, we apply tokenizaition
    y_train.append(tags.index(tag)) # Here we apply the label
                                    # CrossEntropyLoss
    # Ex: ['delivery','funny','goodbye'] it will be labelled as 'delivery': 0, 'funny': 1, 'goodbye': 2

# Convert to numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)   

# Create a new dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()
    
    # This function helps access dataset[idx]
    def __getitem__(self,idx):
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return len(self.x_data)
    
# PyTorch and Training Loop
# Hyperparameters
batch_size = 8
hidden_size= 16
output_size= len(tags)
input_size = len(X_train[0])
learning_rate= 0.001
num_epochs = 1000

train_loader = DataLoader(ChatDataset(), batch_size = batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the Data
for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words = words.to(device)
        labels=labels.to(device)
        
        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
        
print(f'final loss, loss={loss.item():.4f}')

# SAVE THE DATA

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE) # save it to the pickled file?

print(f'training complete. file saved to {FILE}')
