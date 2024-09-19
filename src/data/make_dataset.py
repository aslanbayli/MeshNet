import torch_geometric
import torch
import tqdm
import sys
sys.path.append('.')
from src.features import build_features as bf

'''
3 types of nodes - devices, streaming events (connections), streaming services
edge_attr = probability of connection (e.g. if we have 4 devices, then 0.25)
edge_label = 0 if no connection, 1 if connection
need to project all node attributes onto the same dimension

node_attr for devices =            make, model, appliance_type
node_attr for streaming events =   bandwidth, duration, band, logtime
node_attr for streaming services = channel, brand, parent, lookup_pattern, primary_domain_only, start_time, end_time
'''

def load_graphs(files, out_dim, is_mesh=False):
    dataset = []

    for file in tqdm.tqdm(files):
        graph = bf.build_features(file, is_mesh, out_dim)
        dataset.append(graph)

    graph_dataset = GraphDataset(dataset)
    
    return graph_dataset


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.__data = data

    def len(self):
        return len(self.__data)
    
    def get(self, idx):
        x = self.__data[idx].x
        edge_index = self.__data[idx].edge_index
        edge_attr = self.__data[idx].edge_attr
        y = self.__data[idx].y

        # maintain a parallel array of original indices of the data
        indices = []

        # select only positive edges
        mask = y == 1
        pos_y = y[mask]        
        pos_edge_index = edge_index[:, mask]  
        pos_edge_attr = edge_attr[mask]  

        # number of positive edges 
        pos_size = pos_y.shape[0]  

        # get the indices of the positive edges
        temp_neg_indices = []
        for i, val in enumerate(mask):
            if val == True:
                indices.append(i) # indeces of positive edges
            else:
                temp_neg_indices.append(i) # indeces of negative edges

        # inverse the mask
        mask = ~mask
        neg_y = y[mask]
        neg_edge_index = edge_index[:, mask]
        neg_edge_attr = edge_attr[mask]

        # randomly select from negative edges equal to the number of positive edges
        rand_indices = torch.randint(0, neg_y.shape[0], (pos_size,))
        neg_y = neg_y[rand_indices]
        neg_edge_index = neg_edge_index[:, rand_indices]
        neg_edge_attr = neg_edge_attr[rand_indices]

        # get the indices of the negative edges
        for val in rand_indices:
            indices.append(temp_neg_indices[val])

        # concatenate positive and negative edges 
        new_y = torch.cat([pos_y, neg_y], dim=0)
        new_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        new_edge_attr = torch.cat([pos_edge_attr, neg_edge_attr], dim=0)

        # shuffle the edges
        perm = torch.randperm(new_y.shape[0])
        new_y = new_y[perm]
        new_edge_index = new_edge_index[:, perm]
        new_edge_attr = new_edge_attr[perm]

        # shuffle the indices in the same order
        updated_indices = []
        for val in perm:
            updated_indices.append(indices[val])
 
        # create graph
        graph = torch_geometric.data.Data(
            x = x, 
            edge_index = new_edge_index, 
            edge_attr = new_edge_attr, 
            y = new_y
        )

        return graph, updated_indices


if __name__ == '__main__':  
    is_mesh = sys.argv[1] == 'mesh'
    data = load_graphs([1,2,3,6,7,8,9,10], 100, is_mesh)
    data_loader = torch_geometric.loader.DataLoader(data, batch_size=1, shuffle=False)

    for batch in data_loader:
        pass

