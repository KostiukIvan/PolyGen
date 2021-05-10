import torch.nn as nn

import data_utils.dataloader as dl
from data_utils.transformations import *
from torch.utils.data import DataLoader
from models.VertexModel import VertexModel
from reformer_pytorch import Reformer
from config import VertexConfig
import os

EPOCHS = 10
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

training_data = dl.VerticesDataset(root_dir=dataset_dir,
                                   transform=[SortVertices(),
                                              NormalizeVertices(),
                                              QuantizeVertices(),
                                              ToTensor(),
                                              ResizeVertices(800),
                                              ResizeVertices(801)],
                                   split='train',
                                   classes=None,
                                   train_percentage=0.925)
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

decoder = Reformer(**config['reformer']).to(device)
model = VertexModel(decoder,
                    embedding_dim=config['reformer']['dim'],
                    quantization_bits=8,
                    class_conditional=False,
                    max_num_input_verts=250
                    ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=config['tokenizer']['pad_id'])

if __name__ == "__main__":
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)
            if np.isnan(loss.item()):
                print(f"(E): Model return loss {loss.item()}")
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: loss {total_loss}")
