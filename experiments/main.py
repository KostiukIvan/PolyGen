import numpy as np
import torch.nn as nn

import data_utils.dataloader as dl
from data_utils.transformations import *
from data_utils.transformations import detokenize, extract_vert_values_from_tokens
from data_utils.visualisation import plot_results
from torch.utils.data import DataLoader
from models.VertexModel import VertexModel
from reformer_pytorch import Reformer
from config import VertexConfig
import os

EPOCHS = 1000
GPU = True
dataset_dir = os.path.join(os.getcwd(), 'data', 'shapenet_samples')
config = VertexConfig(embed_dim=256, reformer__depth=6,
                      reformer__lsh_dropout=0.,
                      reformer__ff_dropout=0.,
                      reformer__post_attn_dropout=0.)

if GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = None
"""
training_data = dl.VerticesDataset(root_dir=r"C:\Users\ivank\UJ\Deep learining with multiple tasks\projects\PolyGen\data\shapenet_samples\\",
                                   transform=[SortVertices(),
                                        NormalizeVertices(),
                                        QuantizeVertices(),
                                        ToTensor(),
                                        #ResizeVertices(600),
                                        VertexTokenizer(2400)],
                                   split='train',
                                   classes="02691156",
                                   train_percentage=0.925)
"""
training_data = dl.MeshesDataset("./meshes")

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
decoder = Reformer(**config['reformer']).to(device)
model = VertexModel(decoder,
                    embedding_dim=config['reformer']['dim'],
                    quantization_bits=8,
                    class_conditional=False,
                    max_num_input_verts=1000,
                    device=device
                    ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=VertexTokenizer().tokens['pad'][0].item())

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            data = batch
            target = batch['vertices_tokens'].to(device)
            out = model(data)

            if epoch % 10 == 0:
                sample = np.array([extract_vert_values_from_tokens(sample, seq_len=2400).numpy() for sample in out.cpu()])
                plot_results(sample, f"objects_{epoch}.png")
            out = torch.transpose(out, 1, 2)
            loss = loss_fn(out, target)
            if np.isnan(loss.item()):
                print(f"(E): Model return loss {loss.item()}")
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: loss {total_loss}")
