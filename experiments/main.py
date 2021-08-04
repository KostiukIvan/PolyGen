import os
import time
import numpy as np
import torch.nn as nn

# pip install pytorch3d`
# For me in order to install torch downgrade was required: pip install "torch<1.7"
from pytorch3d.loss import chamfer_distance

import data_utils.dataloader as dl
from data_utils.transformations import *
from data_utils.visualisation import plot_results
from data_utils.tokenizer_vm import VertexTokenizer
from torch.utils.data import DataLoader
from models.VertexModel import VertexModel
from reformer_pytorch import Reformer
from config import VertexConfig
from utils.args import get_args
from utils.checkpoint import load_model
from utils.metrics import accuracy


use_tensorboard = False

EPOCHS = 2500
batch_size = 10
back_prop_freq = 2
save_weights_nth_epoch = 20
seq_len = 2400
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


training_data = dl.VerticesDataset(root_dir="/shared/results/ikostiuk/datasets/polygen_exports",
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

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8)

decoder = Reformer(**config['reformer']).to(device)
tokenizer = VertexTokenizer()
model = VertexModel(decoder,
                    tokenizer,
                    embedding_dim=config['reformer']['dim'],
                    quantization_bits=8,
                    class_conditional=True,
                    num_classes=4,
                    max_num_input_verts=1000,
                    device=device
                    ).to(device)
learning_rate = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
ignore_index = tokenizer.tokens['pad'][0].item()


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
        start = time.time()
        total_loss = total_chamfer_loss = total_accuracy = 0.0
        sample = None
        batch_loss = None
        for i, (batch_d, class_idx) in enumerate(train_dataloader):
            batch_d = {
                    k: v.to(device) for k, v in batch_d.items()
                }
            if i % back_prop_freq == 0 or i == len(train_dataloader) - 1:
                optimizer.zero_grad()
                model_output = model(batch_d)
                loss = model.backward(model_output, batch_d['target_vertices'])
            else:
                model_output = model(batch_d)
                loss+= model.backward(model_output, batch_d['target_vertices'])

            if i % back_prop_freq == 0 or i == len(train_dataloader) - 1:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            # note: remove class embedding when loss is calculated
            # TODO fox for class embedding
            # out = out[:, 1:, :]

            # note: Calculate metrics
            with torch.no_grad():
                out_verts = torch.Tensor([tokenizer.detokenize(sample, seq_len=seq_len).numpy()
                                         for sample in model_output.cpu()]).to(device)
                target_verts = torch.Tensor([tokenizer.detokenize(sample, seq_len=seq_len, is_target=True).numpy()
                                            for sample in batch_d['target_vertices'].cpu()]).to(device)
                total_chamfer_loss += chamfer_distance(out_verts, target_verts)[0].item()

                total_accuracy += accuracy(model_output, batch_d['target_vertices'], ignore_index=ignore_index, device=device)

        if use_tensorboard and i == 0:
            sample = np.array([tokenizer.detokenize(sample, seq_len=seq_len).numpy() for sample in model_output.cpu()])
        elif epoch % 1 == 0:
            recon = np.array([tokenizer.detokenize(sample, seq_len=seq_len).numpy() for sample in model_output.cpu()])
            plot_results(recon, f"reconstruction_{epoch}_{i}.png", output_dir="results")

        print(f"Epoch {epoch}: [{round((time.time() - start) / 60, 2)}]: loss {round(total_loss, 4)}, chamfer_dist {round(total_chamfer_loss, 4)},"
              f" Mean accuracy {round(torch.mean(total_accuracy).item(), 4)}")
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
            writer.add_scalar("Chamfer_distance", total_chamfer_loss, epoch)
            writer.add_scalar("Mean accuracy", torch.mean(total_accuracy), epoch)
            writer.add_mesh('reconstruction', np.interp(sample, (sample.min(), sample.max()), (-0.1, 0.1)), global_step=epoch)
