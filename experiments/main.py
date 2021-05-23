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
use_tensorboard = True

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
training_data = dl.VerticesDataset(root_dir="",
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
batch_size = 1
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
decoder = Reformer(**config['reformer']).to(device)
model = VertexModel(decoder,
                    embedding_dim=config['reformer']['dim'],
                    quantization_bits=8,
                    class_conditional=True,
                    num_classes=4,
                    max_num_input_verts=1000,
                    device=device
                    ).to(device)
learning_rate = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=VertexTokenizer().tokens['pad'][0].item())

writer = None
if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(comment=f'batch_size = {batch_size} learning_rate = {learning_rate}')


if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        sample = None
        for i, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            data, class_idx = batch
            target = data['vertices_tokens'].to(device)
            out = model(data, targets=class_idx,
                        top_p=config['top_p'])

            if use_tensorboard and i == 0:
                sample = np.array([extract_vert_values_from_tokens(sample, seq_len=2400).numpy() for sample in out.cpu()])
            elif epoch % 5 == 0:
                sample = np.array([extract_vert_values_from_tokens(sample, seq_len=2400).numpy() for sample in out.cpu()])
                plot_results(sample, f"objects_{epoch}_{i}.png", output_dir="results_class_embv2")
            # note: remove class embedding when loss is calculated
            out = torch.transpose(out, 1, 2)[:, :, 1:]
            loss = loss_fn(out, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: loss {total_loss}")
        if use_tensorboard:
            writer.add_scalar("Loss/train", total_loss, epoch)
            writer.add_mesh('reconstruction', np.interp(sample, (sample.min(), sample.max()), (-0.1, 0.1)), global_step=epoch)
