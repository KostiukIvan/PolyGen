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

EPOCHS = 2000
GPU = True
dataset_dir = os.path.join(os.getcwd(), 'data', 'shapenet_samples')
config = VertexConfig(embed_dim=128, reformer__depth=6,
                      reformer__lsh_dropout=0.,
                      reformer__ff_dropout=0.,
                      reformer__post_attn_dropout=0.)

if GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = None
"""
training_data = dl.VerticesDataset(root_dir=dataset_dir,
                                   transform=[SortVertices(),
                                              NormalizeVertices(),
                                              QuantizeVertices(),
                                              ToTensor(),
                                              ResizeVertices(799),
                                              ResizeVertices(800)],
                                   split='train',
                                   classes=None,
                                   train_percentage=0.925)
"""
training_data = dl.MeshesDataset("../meshes")
train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
decoder = Reformer(**config['reformer']).to(device)
model = VertexModel(decoder,
                    embedding_dim=config['reformer']['dim'],
                    quantization_bits=8,
                    class_conditional=False,
                    max_num_input_verts=750
                    ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()

if __name__ == "__main__":
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            data = batch[0].to(device)
            out = model(data)

            if epoch % 100 == 0:
                sample = out[0].cpu()
                sample = extract_vert_values_from_tokens(sample)
                plot_results(sample, f"object_{epoch}.png")
            
            loss = loss_fn(out[:, :, 0], data[:, 0])
            if np.isnan(loss.item()):
                print(f"(E): Model return loss {loss.item()}")
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: loss {total_loss}")
