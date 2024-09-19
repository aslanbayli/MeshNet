import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch.nn as nn


class Baseline(nn.Module):
  def __init__(self, edge_attr_size, hidden_size):
    super(Baseline, self).__init__()

    self.edge_proj = nn.Linear(edge_attr_size, hidden_size)
    self.fully_connected_1 = nn.Linear(hidden_size, hidden_size)
    self.fully_connected_2 = nn.Linear(hidden_size, hidden_size)
    self.fully_connected_3 = nn.Linear(hidden_size, hidden_size)
    self.fully_connected_4 = nn.Linear(hidden_size, 1)

  def forward(self, data):
    edge_attr = self.edge_proj(data.edge_attr)
    edge_attr = F.relu(self.fully_connected_1(edge_attr))
    edge_attr = F.relu(self.fully_connected_2(edge_attr))
    edge_attr = F.relu(self.fully_connected_3(edge_attr))
    edge_attr = torch.sigmoid(self.fully_connected_4(edge_attr))

    return edge_attr


# Graph Neural Network
class GraphNN(nn.Module):
    def __init__(self, node_attr_size, edge_attr_size, hidden_size, dropout_prob=0.05):
        super(GraphNN, self).__init__()

        self.node_proj = nn.Linear(node_attr_size, hidden_size)
        self.edge_proj = nn.Linear(edge_attr_size, hidden_size)

        self.node_conv_1 =  pyg_nn.TransformerConv(in_channels=hidden_size, out_channels=hidden_size, edge_dim=hidden_size)
        self.node_conv_2 =  pyg_nn.TransformerConv(in_channels=hidden_size, out_channels=hidden_size, edge_dim=hidden_size)
        self.node_conv_3 =  pyg_nn.TransformerConv(in_channels=hidden_size, out_channels=hidden_size, edge_dim=hidden_size)

        self.edge_update_1 = nn.Linear(3 * hidden_size, hidden_size)
        self.edge_update_2 = nn.Linear(3 * hidden_size, hidden_size)
        self.edge_update_3 = nn.Linear(3 * hidden_size, hidden_size)

        self.fully_connected_1 = nn.Linear(hidden_size, 1)

        self.dropout_1 = nn.Dropout(dropout_prob)
        self.dropout_2 = nn.Dropout(dropout_prob)

    def forward(self, data):
        x = self.node_proj(data.x)
        edge_attr = self.edge_proj(data.edge_attr)

        # first layer
        x = self.node_conv_1(x, data.edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.dropout_1(x)
        edge_attr = self.edge_update_1(torch.cat([x[data.edge_index[0]], x[data.edge_index[1]], edge_attr], dim=-1))
        edge_attr = F.leaky_relu(edge_attr)
        edge_attr = self.dropout_1(edge_attr)

        # second layer
        x = self.node_conv_2(x, data.edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.dropout_2(x)
        edge_attr = self.edge_update_2(torch.cat([x[data.edge_index[0]], x[data.edge_index[1]], edge_attr], dim=-1))
        edge_attr = F.leaky_relu(edge_attr)
        edge_attr = self.dropout_2(edge_attr)

        # third layer
        x = self.node_conv_3(x, data.edge_index, edge_attr)
        edge_attr = self.edge_update_3(torch.cat([x[data.edge_index[0]], x[data.edge_index[1]], edge_attr], dim=-1))

        # fully connected layer
        edge_attr = F.sigmoid(self.fully_connected_1(edge_attr))

        return edge_attr