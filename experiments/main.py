from torch.utils.data import DataLoader
from reformer_pytorch import Reformer
import torch
import numpy as np

import data_utils.dataloader as dl
from models.VertexModel_test import VertexPolyGen
from experiments.config import VertexConfig



def detokenize(vertices_tokens):
    return torch.reshape(vertices_tokens, shape=(-1, 3))

def extract_vert_values_from_tokens(vert_tokens):
    vert_tokens = torch.max(vert_tokens, dim=1)[1]
    vertices = detokenize(vert_tokens)
    vertices = vertices.float()
    vertices /= 256
    return vertices
    
    
import os
import matplotlib.pyplot as plt


def plot_results(vertices, filename, *, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    ax.scatter(x, y, z)
    plt.savefig(os.path.join(output_dir, filename))



EPOCHS = 2000
GPU = True
batch_size = 1
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
for i in range(0, len(training_data)):
    plot_results(training_data[i], f"real_{i}.png")

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
        loss, acc, out = model(batch, device)

        if epoch % 10 == 0:
            sample = out[0:2400,:].cpu()
            sample = extract_vert_values_from_tokens(sample)
            plot_results(sample, f"object_{epoch}.png")
        

        if np.isnan(loss.item()):
            print(f"(E): Model return loss {loss.item()}")
        total_loss += loss.item()
        total_acc += acc.item() 
        amount += batch_size
        loss.backward()
        optimizer.step()
    print(f"\nEpoch : {epoch}, loss : {total_loss / amount}, acc : {total_acc / amount}")

