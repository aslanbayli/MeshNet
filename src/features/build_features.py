import numpy as np
import pandas as pd
import torch
import torch_geometric
import torch.nn.functional as F
from sklearn.feature_extraction.text import HashingVectorizer
import sys
sys.path.append('.')
from src.models import utils
from datetime import timedelta
from tqdm import tqdm


def combine_dataset(h_num, is_mesh):
    data_file = './data/grouped_by_hh_light.xlsx'
    non_mesh = [234562, 564366, 1029485, 3543200, 3947624, \
                4531775, 4645673, 5643645, 6786786, 8978943, 23467890] # 11 households
    mesh = [2356192, 3583772, 6273845, 9900111] # 4 households

    # retrieve the household number
    hh = ''
    if is_mesh:
        hh = str(mesh[h_num])
    else:
        hh = str(non_mesh[h_num])

    # define the sheet names
    sheet_ce = f'hh-{hh}-ce'
    sheet_be = f'hh-{hh}-be'
    if is_mesh:
        sheet_ce = f'hh-{hh}-mesh-ce'
        sheet_be = f'hh-{hh}-mesh-be'

    # read in raw data
    raw_data_ce = pd.read_excel(data_file, index_col=None, header=0, sheet_name=sheet_ce, engine='openpyxl', dtype=str)
    raw_data_be = pd.read_excel(data_file, index_col=None, header=0, sheet_name=sheet_be, engine='openpyxl', dtype=str)

    # replace all empty values with 'NULL'
    raw_data_ce.fillna('NULL', inplace=True)
    raw_data_be.fillna('NULL', inplace=True)

    # define time thresholds
    start_time_threshold = 10
    s_th = timedelta(minutes=start_time_threshold)
    end_time_threshold = 5
    e_th = timedelta(minutes=end_time_threshold)

    # index mapping between ce and be sheet by time
    map_ce_be = {}
    for idx, (m1, s, e) in enumerate(zip(raw_data_ce['Obfuscated MAC'].to_numpy(), raw_data_ce['starttime_est'].to_numpy(),raw_data_ce['endtime_est'].to_numpy())):
        if s == 'NULL' or e == 'NULL':
            continue

        s = utils.get_datetime(s)
        e = utils.get_datetime(e)
        matches = []
        for idx2, (m2, l, b) in enumerate(zip(raw_data_be['Obfuscated MAC'].to_numpy(), raw_data_be['logtime'].to_numpy(), raw_data_be['bandwidth'].to_numpy())):
            if m2 != m1:
                continue

            # discard bandwidths with only 1's and 0's or with single value
            b_arr = b.split(',')
            ones = b_arr.count('1')
            zeros = b_arr.count('0')
            ones_and_zeros = ones + zeros
            if ones_and_zeros == len(b_arr) or len(b_arr) <= 1:
                continue
            
            if l == 'NULL':
                continue
            l = utils.get_datetime(l)

            edge_weight = 0
            diff = 0
            if l >= s and l <= e:
                edge_weight = 1
            elif l < s and l >= s - s_th:
                diff = s - l
            elif l > e and l <= e + e_th:
                diff = l - e

            if diff != 0:
                time_str = str(diff).split(':')
                hours = int(time_str[0]) * 3600
                minutes = int(time_str[1]) * 60
                seconds = int(time_str[2])
                diff_s = hours + minutes + seconds # time difference in seconds
                edge_weight = utils.calc_edge_weight(end_time_threshold, diff_s)
                matches.append( (idx2, edge_weight, 1) )
            
        if len(matches) > 0:
            map_ce_be[idx] = matches
        
    # put all data into a list
    data = []
    for key, val in map_ce_be.items():
        # get row at index ce_idx from dataframe raw_data_ce
        ce_idx = key
        ce_row = raw_data_ce.iloc[ce_idx]

        # get row at index be_idx from dataframe raw_data_be
        for idx, (be_idx, prob, e_type) in enumerate(val):
            be_row = raw_data_be.iloc[be_idx]
            # select wanted columns from be_row
            be_row = be_row[["duration", "logtime", "band", "bandwidth"]]

            # temp represents a row of data that will be in the output file
            temp = np.concatenate((ce_row.values, be_row.values, [prob, e_type]))
            data.append(temp)

    # create a pandas dataframe for saving to file
    df = pd.DataFrame(columns=["Obfuscated hhid", "Obfuscated MAC", "primarydomainonly", \
                                "make", "model", "appliance_type", "channel", "brand", "parent", \
                                "lookup_pattern", "bandwidth", "starttime_est", "endtime_est", \
                                "duration", "logtime", "band", "bandwidth_from_be", "connection_probability", "edge_type"], data=data)
 
    # drop the index column
    df.reset_index(drop=True, inplace=True)      

    return df


def create_dataset(h_num, is_mesh):
    data = combine_dataset(h_num, is_mesh)

    device_node_attr = data[['make', 'model', 'appliance_type']].to_numpy()
    streaming_node_attr = data[['edge_type', 'connection_probability', 'logtime', 'starttime_est', \
                                'endtime_est', 'bandwidth_from_be', 'duration', 'band', \
                                'channel', 'brand', 'parent', 'primarydomainonly', 'lookup_pattern']].to_numpy()

    ### remove duplicates ###
    delimiter = '--' # delimiter for joining strings
    unique_devices = {}
    for idx, d in enumerate(device_node_attr):
        key = delimiter.join(d)
        if key not in unique_devices:
            unique_devices[key] = [idx]
        else:
            unique_devices[key].append(idx)

    # normalize connection probability so they add up to 1 for the same device
    for key, val in unique_devices.items():
        prob = []
        for i in val:
            prob.append(streaming_node_attr[i][1])
        s = sum(prob)
        normalized_prob = [float(i)/s for i in prob]
        for i, idx in enumerate(val):
            streaming_node_attr[idx][1] = normalized_prob[i]

    return unique_devices, streaming_node_attr


def save_to_file(is_mesh):
    out_file = './reports/processed/grouped_by_hh_combined.xlsx'

    if is_mesh:
        for i in tqdm(range(0, 5)):
            df = combine_dataset(i, True)
            with pd.ExcelWriter(out_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                # get the column named 'Obfuscated hhid'
                hh = df['Obfuscated hhid'].to_numpy()[0]
                # filter out rows where edge_type is 0
                df = df[df['edge_type'] == 1]
                df.to_excel(writer, sheet_name=f'hh-{hh}-mesh-combined', index=False)
    else:
        for i in tqdm(range(0, 11)):
            df = combine_dataset(i, False)
            with pd.ExcelWriter(out_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                # get the column named 'Obfuscated hhid'
                hh = df['Obfuscated hhid'].to_numpy()[0]
                # filter out rows where edge_type is 0
                df = df[df['edge_type'] == 1]
                df.to_excel(writer, sheet_name=f'hh-{hh}-combined', index=False)
            

def build_features(h_num, is_mesh, out_dim):
    unique_devices_map, streaming_node_attr = create_dataset(h_num, is_mesh)
    edge_index = []
    edge_attr = []
    edge_label = []

    num_devices = len(unique_devices_map)
    num_rows = streaming_node_attr.shape[0]

    # add all possible edge indices device -> streaming
    for idx, (_, val) in enumerate(unique_devices_map.items()):
        for i in range(0, num_rows):
            edge_index.append([idx, i+num_devices])
            edge_attr.append([streaming_node_attr[i][1]]) # connection probability
            if i in val:
                edge_label.append(1)
            else:
                edge_label.append(0)

    # define a hashing function for string tokenization
    h_vectorizer = HashingVectorizer(n_features=100, norm=None, alternate_sign=False)

    new_device_node_attr = []
    for key, val in unique_devices_map.items():    
        row = key.split('--')    
        row = h_vectorizer.fit_transform(row).toarray()
        row = row.flatten()        
        # normalize the values
        res = utils.z_score_normalize(row)
        new_device_node_attr.append(row)
    
    new_streaming_node_attr = []
    for row in streaming_node_attr:
        # logtime
        logtime = row[2]
        h, m, _ = utils.one_hot_encode_time(logtime)
        # calculate duration
        s = row[3]
        e = row[4]
        h2, m2, _ = utils.one_hot_encode_duration(s, e)
        # badnwidth statistics
        bandwidth = row[5]
        b = utils.bandwidth_stats(bandwidth)
        # duration, band
        numericals = np.array([float(i) for i in row[6:8]])
        # tokenize the rest of the cols
        tokens = row[8:]
        tokens = h_vectorizer.fit_transform(tokens).toarray()
        tokens = tokens.flatten()
        # concatenate all
        res = np.concatenate((h, m, h2, m2, b, numericals, tokens), axis=None)
        # normalize the values
        res = utils.z_score_normalize(res)
        new_streaming_node_attr.append(res)

    # convert to numpy arrays
    new_device_node_attr = np.array(new_device_node_attr)
    new_streaming_node_attr = np.array(new_streaming_node_attr)
    edge_attr = np.array(edge_attr)

    # convert all node attributes to torch tensors
    new_device_node_attr = torch.tensor(new_device_node_attr, dtype=torch.float)
    new_streaming_node_attr = torch.tensor(new_streaming_node_attr, dtype=torch.float)

    # project the node attributes to the same dimension
    with torch.no_grad():
        new_device_node_attr = F.linear(new_device_node_attr, torch.randn(out_dim, new_device_node_attr.shape[1]))
        new_streaming_node_attr = F.linear(new_streaming_node_attr,  torch.randn(out_dim, new_streaming_node_attr.shape[1]))

    # convert to torch tensors
    node_attr = torch.cat((new_device_node_attr, new_streaming_node_attr), dim=0)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_label = torch.tensor(edge_label, dtype=torch.float)

    # create graph
    graph = torch_geometric.data.Data(
        x = node_attr, 
        edge_index = edge_index, 
        edge_attr = edge_attr, 
        y = edge_label
    )

    return graph    


if __name__ == '__main__':
    # save_to_file(False)
    build_features(1, False, 100)
    # for i in range(0, 11):
    # #     create_dataset(i, False)
    # for i in range(0, 11):
    #     combine_dataset(i, False)
