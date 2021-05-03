import data_utils.dataloader as dl
from data_utils.transformations import *
from torch.utils.data import DataLoader


training_data = dl.VerticesDataset(root_dir=r'C:\Users\ivank\UJ\Deep learining with multiple tasks\projects\ShapeNetCore_PolyGenSubset\train',
                classes=['basket'], transform=[SortVertices(), NormalizeVertices(), QuantizeVertices(), ToTensor(),
                                               ResizeVertices(500)], split='train', train_percentage=0.925)

train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)


for i, vertices in enumerate(train_dataloader):
    print(i)










