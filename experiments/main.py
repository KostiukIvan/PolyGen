import os
import numpy as np
import torch.nn as nn

import data_utils.dataloader as dl
from data_utils.transformations import *
from data_utils.visualisation import plot_results
from torch.utils.data import DataLoader
from models.VertexModel import VertexModel
from reformer_pytorch import Reformer
from config import VertexConfig
from utils.args import get_args
from utils.checkpoint import load_model


use_tensorboard = False

EPOCHS = 1000
batch_size = 6
back_prop_freq = 1
save_weights_nth_epoch = 50
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


training_data = dl.VerticesDataset(root_dir="/mnt/remote/wmii_gmum_projects/datasets/ShapeNetPolygen/",
                                   transform=[SortVertices(),
                                        NormalizeVertices(),
                                        QuantizeVertices(),
                                        ToTensor(),
                                        #ResizeVertices(600),
                                        VertexTokenizer(2400)],
                                   split='train',
                                   classes="02691156",
                                   train_percentage=0.99)
'''
training_data = dl.MeshesDataset("./meshes")
'''
print("Loaded data len: ", len(training_data))

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
decoder = Reformer(**config['reformer']).to(device)
model = VertexModel(decoder,
                    embedding_dim=config['reformer']['dim'],
                    quantization_bits=8,
                    class_conditional=True,
                    num_classes=1,
                    max_num_input_verts=1000,
                    device=device
                    ).to(device)
learning_rate = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
ignore_index = VertexTokenizer().tokens['pad'][0].item()
loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

writer = None
if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(comment=f'batch_size = {batch_size} learning_rate = {learning_rate}')

if __name__ == "__main__":
    params = get_args()

    # Load weights if provided param or create dir for saving weights
    # e.g. -load_weights path/to/weights/epoch_x.pt
    model_weights_path, epoch, total_loss = load_model(params, model, optimizer)
    model.train()
    for epoch in range(epoch, EPOCHS + 1):
        total_loss = 0.0
        sample = None
        batch_loss = None
        for i, batch in enumerate(train_dataloader):
            if i == len(train_dataloader) or (i + 1) % back_prop_freq == 0:
                optimizer.zero_grad()

            data, class_idx = batch
            target = data['vertices_tokens'].to(device)
            
            out = model(data, targets=class_idx)
            #print(out.shape, data['vertices_tokens'].shape)
            # note: remove class embedding when loss is calculated
            out = torch.transpose(out, 1, 2)#[:, :, 1:]
            loss = loss_fn(out, target)
            total_loss += loss.item()

            if batch_loss:
                batch_loss += loss.item()
            else:
                batch_loss = loss
            if i == len(train_dataloader) or (i + 1) % back_prop_freq == 0:
                batch_loss.backward()
                optimizer.step()
                batch_loss = None

        if use_tensorboard and i == 0:
            sample = np.array([extract_vert_values_from_tokens(sample, seq_len=2400).numpy() for sample in out.cpu()])
        elif epoch % 1 == 0:
            recon = np.array([extract_vert_values_from_tokens(sample, seq_len=2400).numpy() for sample in out.cpu()])
            plot_results(recon, f"reconstruction_{epoch}_{i}.png", output_dir="results_class_embv2")

        print(f"Epoch {epoch}: loss {total_loss}")
        if epoch % save_weights_nth_epoch == 0:
            print('saving weights')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, os.path.join(model_weights_path, 'epoch_' + str(epoch) + '.pt'))
        if use_tensorboard:
            writer.add_scalar("Loss/train", total_loss, epoch)
            writer.add_mesh('reconstruction', np.interp(sample, (sample.min(), sample.max()), (-0.1, 0.1)), global_step=epoch)
