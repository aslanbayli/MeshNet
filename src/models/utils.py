import torch
import numpy as np
from datetime import datetime
import pandas as pd


def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def z_score_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data


def one_hot_encode_time(time_str):
    # convert to datetime object
    date_obj = get_datetime(time_str)
    
    # access components
    hour = date_obj.hour
    minute = date_obj.minute
    second = date_obj.second

    # create an array of size 24 for stroing one-hot encoded hour
    hour_arr = np.zeros((24,), dtype=float)
    hour_arr[hour] = 1.0

    # create an array of size 60 for storing one-hot encoded minute
    minute_arr = np.zeros((60,), dtype=float)
    minute_arr[minute] = 1.0

    # create an array of size 60 for storing one-hot encoded second
    second_arr = np.zeros((60,), dtype=float)
    second_arr[second] = 1.0

    return hour_arr, minute_arr, second_arr


def one_hot_encode_duration(start_time_str, end_time_str):
    # convert to datetime object
    start_date_obj = get_datetime(start_time_str)
    end_date_obj = get_datetime(end_time_str)

    # access components
    start_hour = start_date_obj.hour
    start_minute = start_date_obj.minute
    start_second = start_date_obj.second

    end_hour = end_date_obj.hour
    end_minute = end_date_obj.minute
    end_second = end_date_obj.second

    # create an array of size 24 for stroing one-hot encoded hour difference
    hour_arr = np.zeros((24,), dtype=float)
    hour_arr[start_hour:end_hour+1] = 1.0

    # create an array of size 60 for storing one-hot encoded minute difference
    minute_arr = np.zeros((60,), dtype=float)
    minute_arr[start_minute:end_minute+1] = 1.0

    # create an array of size 60 for storing one-hot encoded second difference
    second_arr = np.zeros((60,), dtype=float)
    second_arr[start_second:end_second+1] = 1.0

    return hour_arr, minute_arr, second_arr


def bandwidth_stats(bandwidth_str):
    # split the string into a list of values
    values = bandwidth_str.split(',')
    
    # remove the last value if it ends with '...'
    if values[-1].endswith('...'):
        values = values[:-1]

    # filter out non-numeric values and convert to integers
    values = [int(v) for v in values if v.isdigit()]

    # calculate average, standard deviation, minimum, and maximum
    avg = sum(values) / len(values) if len(values) > 1 else 0
    std_dev = pd.Series(values).std() if len(values) > 1 else 0
    min_val = pd.Series(values).min() if len(values) > 1 else 0
    max_val = pd.Series(values).max() if len(values) > 1 else 0

    return np.array([float(avg), float(std_dev), float(min_val), float(max_val)])


def calc_edge_weight(threshold, x):
    '''
    y = - 1/(t+0.1)x + 1
    '''
    th_s = threshold * 60 # threshold in seconds
    m = -1/(th_s + 0.1) # slope
    y = m * x + 1
    return y


def helper(dict, val1, val2):
    if len(val1) < 5:
        return dict[val2]
    else: 
        return val1
    

def fill_mac(truth="truth"):
    #read csv
    truth_df= pd.read_csv(f"./src/experiments/{truth}.csv", header=0)

    #strip whitespaces from header column
    truth_df.columns = truth_df.columns.str.strip()
    truth_df['MAC Address'] = truth_df['MAC Address'].astype(str)

    #replace empty MAC Address based on label data
    map_truth_dict = {}
    for i,j in zip(truth_df['MAC Address'].values.tolist(),truth_df['Label'].values.tolist()):
        if j not in map_truth_dict:
            if 'NULL' not in j and 'nan' not in i:
                map_truth_dict[j] = i
    print(map_truth_dict)
    
    truth_df['MAC Address'] = truth_df.apply(lambda x: helper(map_truth_dict,x['MAC Address'], x['Label']), axis=1)


def compare_csv(hh="1029485"):
    #read csv
    df_1 = pd.read_csv(f"./reports/processed/hh_{hh}_map.csv", header=0)
    sheet = f'hh-{hh}-combined'
    df_2 = pd.read_excel('./reports/processed/grouped_by_hh_combined.xlsx', index_col=None, header=0, sheet_name=sheet, engine='openpyxl', dtype=str)

    #select same columns as in prediction
    df_2 = df_2[['make', 'model', 'appliance_type', 'logtime', 'starttime_est', 'endtime_est']]
    
    #replace NaN's will NULL
    df_1.fillna('NULL', inplace=True)
    df_2.fillna('NULL', inplace=True)

    #combine dataframes
    frames = [df_1, df_2]
    result = pd.concat(frames)

    #true positives
    df_1_TP = pd.DataFrame(result.duplicated(keep=False),columns=["TPs"])
    df_1_TP = result.join(df_1_TP)
    df_1_TP = df_1_TP.drop_duplicates()
    df_1_TP.to_csv(f"./src/experiments/TPs.csv")

    #true positive counts
    count_TP = result.duplicated().sum()
    p = count_TP/(len(df_1) + 1) #Add the header to the count
    print(f"TP Count: {count_TP}, Percentage Correct: {p*100}%")

    #false positives
    df_1_only = set(df_1.apply(tuple, axis=1)) - set(df_2.apply(tuple, axis=1))
    pd.DataFrame(list(df_1_only), columns=df_1.columns).to_csv(f"./src/experiments/FPs.csv")
    count_FP = len(df_1_only)
    print(f"FP Count: {count_FP}")

    #false negatives
    df_2_only = set(df_2.apply(tuple, axis=1)) - set(df_1.apply(tuple, axis=1))
    pd.DataFrame(list(df_2_only), columns=df_2.columns).to_csv(f"./src/experiments/FNs.csv")
    count_FN = len(df_2_only)
    print(f"FN Count: {count_FN}")


def get_datetime(data, idx=0):
    if idx > 0:
        date_str = data[idx]
    else:
        date_str = data

    date_format = '%Y-%m-%d %H:%M:%S'
    return datetime.strptime(date_str, date_format)


def create_mappings(pred, indices, h_num, is_mesh):
    node_attr, edge_index, hh = build_map_features(h_num, is_mesh)
    mapping = []

    stop = int(pred.shape[0]/3*2)
    for i in range(0, stop):
        p = pred[i]
        if p.item() >= 0.5:
            j = indices[i]
            idx_1, idx_2 = edge_index[j][0], edge_index[j][1]
            mapping.append([node_attr[idx_1], node_attr[idx_2]])

    # save mapping to a file    
    with open(f"./reports/processed/hh_{hh}_map.csv", "w") as f:
        f.write("make, model, appliance_type, logtime, starttime_est, endtime_est\n")
        for m in mapping:
            f.write(f'{m[0][0]}, {m[0][1]}, {m[0][2]}, {m[1][0]}, {m[1][1]}, {m[1][2]}\n')


def create_mappings_and_filter(pred, indices, h_num, is_mesh):
    node_attr, edge_index, hh = build_map_features(h_num, is_mesh)
    mapping = []

    stop = int(pred.shape[0]/3*2)
    for i in range(0, stop):
        p = pred[i]
        if p.item() >= 0.5:
            j = indices[i]
            idx_1, idx_2 = edge_index[j][0], edge_index[j][1]
            mapping.append([node_attr[idx_1], node_attr[idx_2]])

    # mapping_processed list
    mapping_processed = []

    # save mapping to a file
    with open(f"./src/experiments/hh_{hh}_filteredmap.csv", "w") as f:
        f.write("MAC Address, Primary Domain, Bandwidth, Model, Applince Type, Channel, Brand, Parent, Start Time, End Time\n")
        for m in mapping:
            f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(m[0][0], m[0][1], m[0][2], m[0][3], m[0][4], m[1][0], m[1][1], m[1][2], m[1][3], m[1][4]))
             # remove null MACs
            if m[0][0] != 'NULL':    
                mapping_processed.append([m[0][0],m[0][1],m[1][4],m[1][1],m[1][2],m[2]])

    # sort records by start_time
    mapping_processed.sort(key=get_datetime(3))

    # removing duplicate records
    mapping_processed = pd.Series(mapping_processed).drop_duplicates().tolist()
    
    # removing duplicate records by probability score
    mapping_processedH = {}
    mapping_processedF = []
    for e in mapping_processed:
        keyCreate = ''.join(e[:-1])
        if keyCreate not in mapping_processedH:
            mapping_processedH[keyCreate] = [e[5],e]
        else:
            if mapping_processedH[keyCreate][0] < e[5]:
                mapping_processedH[keyCreate] = [e[5],e]

    for key in mapping_processedH:
        mapping_processedF.append(mapping_processedH[key][1])

    with open(f"./src/experiments/hh_{hh}_map_clean.csv", "w") as f:
        f.write("MAC Address, Primary Domain, Bandwidth, Model, Applince Type, Channel, Brand, Parent, Start Time, End Time, Probability\n")
        for m in mapping_processedF:
            f.write("{}, {}, {}, {}, {}, {}\n".format(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10]))


def build_map_features(h_num, is_mesh):
    non_mesh = [234562, 564366, 1029485, 3543200, 3947624, \
                4531775, 4645673, 5643645, 6786786, 8978943, 23467890] # 11 households
    mesh = [2356192, 3583772, 6273845, 9900111] # 4 households

    hh = 0
    if is_mesh:
        hh = mesh[h_num]
    else:
        hh = non_mesh[h_num]

    sheet = ''
    if is_mesh:
        sheet = f'hh-{hh}-mesh-combined'
    else:
        sheet = f'hh-{hh}-combined'

    # read in raw data
    data = pd.read_excel('./reports/processed/grouped_by_hh_combined.xlsx', index_col=None, header=0, sheet_name=sheet, engine='openpyxl', dtype=str)

    device_node_attr = data[['make', 'model', 'appliance_type']].to_numpy()
    streaming_node_attr = data[['logtime', 'starttime_est', 'endtime_est']].to_numpy()

    ### remove duplicates ###
    delimiter = '--' # delimiter for joining strings
    unique_devices = {}
    for idx, d in enumerate(device_node_attr):
        for i in range(0, len(d)):
            if type(d[i]) is not str:
                d[i] = 'NULL'
        key = delimiter.join(d)
        if key not in unique_devices:
            unique_devices[key] = [idx]
        else:
            unique_devices[key].append(idx)

    edge_index = []
    num_devices = len(unique_devices)
    num_rows = streaming_node_attr.shape[0]

    # add all possible edge indices device -> streaming
    for idx in range(0, num_devices):
        for i in range(0, num_rows):
            edge_index.append([idx, i+num_devices])
    
    # concat device and streaming node attributes
    devices = []
    for key in unique_devices:
        devices.append(device_node_attr[unique_devices[key][0]])
    devices = np.array(devices)
    node_attr = np.concatenate((devices, streaming_node_attr), axis=0)

    return node_attr, edge_index, hh


def overall_accuracy(model, predictions, truth):
    correct = 0
    total = 0

    model.eval() # switch to evaluation mode
    
    with torch.no_grad():
        for i in range(len(predictions)):
            pred = 0
            if predictions[i].item() >= 0.5:
                pred = 1

            if pred == truth[i].item():
                correct += 1
            total += 1

    accuracy = correct / total * 100

    model.train() # switch back to training mode

    return accuracy


def accuracy(model, predictions, truth):
    pos_correct = neg_correct = 0
    pos_count = neg_count = 0

    model.eval() # switch to evaluation mode
    
    with torch.no_grad():
        for i in range(len(predictions)):
            pred = 0
            if predictions[i].item() >= 0.5:
                pred = 1

            if truth[i].item() == 1.0:
                pos_count += 1
                if pred == 1:
                    pos_correct += 1
            else:
                neg_count += 1
                if pred == 0:
                    neg_correct += 1

    pos_accuracy = pos_correct / pos_count * 100
    neg_accuracy = neg_correct / neg_count * 100

    model.train() # switch back to training mode

    return pos_accuracy, neg_accuracy


def confusion_matrix(predictions, truth):
    """
    matrix = [
        true_positives, false_positives,
        false_negatives, true_negatives
    ]
    """
    matrix = np.zeros(4) 
    for i in range(len(predictions)):
        pred = 0
        if predictions[i].item() >= 0.5:
            pred = 1
        
        if pred == 1 and truth[i].item() == 1: # True positive
            matrix[0] += 1  
        elif pred == 1 and truth[i].item() == 0: # False positive
            matrix[1] += 1
        elif pred == 0 and truth[i].item() == 1: # False negative
            matrix[2] += 1
        elif pred == 0 and truth[i].item() == 0: # True negative
            matrix[3] += 1

    return matrix


def f1_score(confusion_matrix):
    tp, fp, fn, _ = confusion_matrix

    epsilon = 1e-7  # A small constant to avoid division by zero
    precision = (tp + epsilon) / (tp + fp + epsilon) * 100
    recall = (tp + epsilon) / (tp + fn + epsilon) * 100
    fscore = 2 * (precision * recall) / (precision + recall) 
    
    return precision, recall, fscore


# use this if you want to assign weights to the loss function
def calc_weights(y):
    pos_count = torch.sum(y == 1).item()
    neg_count = torch.sum(y == 0).item()
    pos_weight = torch.tensor(1 / pos_count )
    neg_weight = torch.tensor(1 / neg_count)
    weights = torch.where(y == 1, pos_weight, neg_weight)

    return weights


# Confidence Based Performance Estimation (CBPE)
def cbpe_confusion_matrix(predictions):
    matrix = np.zeros(4) 

    for i in range(len(predictions)):
        pred = predictions[i].item()
        if pred >= 0.5:
            matrix[0] += pred # True positive
            matrix[1] += 1 - pred # False positive
        else:
            matrix[2] += pred # False negative
            matrix[3] += 1 - pred # True negative
    
    return matrix


if __name__ == '__main__':
    compare_csv()
