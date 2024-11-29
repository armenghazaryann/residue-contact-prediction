import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

from data_prep import ProteinDataset
from protein_loading import protein_ids

class GNNModel(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(node_input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 + edge_input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)

        edge_logits = self.edge_mlp(edge_features).squeeze()
        return edge_logits


def collate_fn(batch):
    return batch


def train(batch_size=8, num_epochs=5) -> None:
    # Prepare the dataset and dataloader with batching
    pdb_files = [f'pdb_files/{i}.pdb' for i in protein_ids[:-1]]
    dataset = ProteinDataset(pdb_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_input_dim = 320 + 1
    edge_input_dim = 1
    hidden_dim = 128

    # Initialize model and optimizer
    model = GNNModel(node_input_dim, edge_input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')
    torch.save(model.state_dict(), 'gnn_model_100_8_5.pth')

if __name__ == '__main__':
    train()