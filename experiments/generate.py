import numpy as np
from data_utils.transformations import *
from models.VertexModel import VertexModel
from data_utils.visualisation import plot_results
from reformer_pytorch import Reformer
from config import VertexConfig
from utils.args import get_args
from utils.checkpoint import load_model

use_tensorboard = True
seq_len = 810

GPU = True
config = VertexConfig(embed_dim=256, reformer__depth=6,
                      reformer__lsh_dropout=0.,
                      reformer__ff_dropout=0.,
                      reformer__post_attn_dropout=0.)

if GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = None

vertex_tokenizer = VertexTokenizer(max_seq_len=810)
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
ignore_index = vertex_tokenizer.tokens['pad'][0].item()

if __name__ == "__main__":
    model.eval()
    params = get_args()
    # Load weights if provided param or create dir for saving weights
    # e.g. -load_weights path/to/weights/epoch_x.pt
    model_weights_path, epoch, total_loss = load_model(params, model, optimizer)
    with torch.no_grad():
        out = model.sample(
            num_samples=1,
            tokenizer=vertex_tokenizer,
            max_sample_length=seq_len,
            top_p=0.9
        ).transpose(1, 0)
    print(out.size())
    sample = np.array([extract_vert_values_from_tokens(sample, seq_len=seq_len, is_target=True).numpy() for sample in out.cpu()])
    plot_results(sample, "objects_gerated.png", output_dir="generate")
