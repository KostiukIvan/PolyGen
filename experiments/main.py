from torch.utils.data import DataLoader
from reformer_pytorch import Reformer
import torch
import numpy as np

import data_utils.dataloader as dl
from models.VertexModel_test import VertexPolyGen
from experiments.config import VertexConfig

EPOCHS = 100
GPU = True
batch_size = 16
config = VertexConfig(embed_dim=128, reformer__depth=6,
                             reformer__lsh_dropout=0.,
                             reformer__ff_dropout=0.,
                             reformer__post_attn_dropout=0.)


if GPU and torch.cuda.is_available():
    print("GPU")
    device = torch.device('cuda')
else:
    device = None

training_data = dl.VerticesDataset(**config['train_dataset'])
print(len(training_data))
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=16)

#decoder = Reformer(**config['reformer'])
model = VertexPolyGen(config, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    total_acc = 0.0
    amount = 1
    for i, batch in enumerate(train_dataloader):
        print("=", end="")
        optimizer.zero_grad()
        loss, acc = model(batch, device)
        if np.isnan(loss.item()):
            print(f"(E): Model return loss {loss.item()}")
        total_loss += loss.item()
        total_acc += acc.item() 
        amount += batch_size
        loss.backward()
        optimizer.step()
    print(f"\nEpoch : {epoch}, loss : {total_loss / amount}, acc : {total_acc / amount}")

