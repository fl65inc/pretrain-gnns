import torch
from tqdm import tqdm
from model import GNN_graphpred
from loader import MoleculeDataset
from torch_geometric.data import DataLoader
from util import ExtractSubstructureContextPair

def predict(num_layer=5, csize=3, mode='cbow', batch_size=1, num_workers=1):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    l1 = num_layer - 1
    l2 = l1 + csize

    print(mode)
    print("num layer: %d l1: %d l2: %d" %(num_layer, l1, l2))

    #set up dataset and transform function.
    dataset = MoleculeDataset("/tmp/dataset", dataset='tox21')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)

    #set up models, one for pre-training and one for context embeddings
    num_tasks = 12
    saved_model = GNN_graphpred(num_layer, 300, num_tasks)
    saved_model.from_pretrained('model_gin/contextpred.pth')
    saved_model.to(device)
    with torch.no_grad():
        for batch in tqdm(loader):
            print(batch)
            batch.to(device)
            
            prediction = saved_model(batch)
            print(prediction)

if __name__ == '__main__':
    predict()