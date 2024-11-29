import torch
from torch_geometric.data import Data, Dataset

from esm2 import get_embeddings
from processing import pdb_extractor


def create_data_object(results: dict, sequence_embeddings):
    embeddings = sequence_embeddings[0, 1:-1, :]

    relative_positions = torch.tensor(
        [prop['Relative Position'] for idx, prop in results['Residue Properties']],
        dtype=torch.float
    ).unsqueeze(1)

    x = torch.cat([embeddings, relative_positions], dim=1)

    n = x.size(0)

    row, col = torch.meshgrid(torch.arange(n), torch.arange(n))
    row, col = row.reshape(-1), col.reshape(-1)
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)

    interaction_matrix = torch.tensor(results['Edge Properties'], dtype=torch.float)
    edge_attr = interaction_matrix.reshape(-1)[mask].unsqueeze(1)

    distance_matrix = torch.tensor(results['Distance matrix'], dtype=torch.float)
    labels = (distance_matrix < 8.0).float()
    edge_labels = labels.reshape(-1)[mask]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)
    return data


class ProteinDataset(Dataset):
    def __init__(self, pdb_files, transform=None):
        super(ProteinDataset, self).__init__()
        self.pdb_files = pdb_files
        self.transform = transform

    def len(self):
        return len(self.pdb_files)

    def get(self, idx):
        pdb_file = self.pdb_files[idx]
        results = pdb_extractor(pdb_file)

        sequence_embeddings = get_embeddings(results["Full Sequence"])
        data = create_data_object(results, sequence_embeddings)

        if self.transform:
            data = self.transform(data)
        return data