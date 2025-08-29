#SAMPLE CODE FOR TRAJECTORY PREDICTION IN THE ETH SCENARIO

import argparse
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import tqdm  
import random
import matplotlib.pyplot as plt
from collections import defaultdict  
from sklearn.cluster import DBSCAN  
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler 


from soft_dtw import SoftDTW  

#===

# Path Configuration

PROJECT_ROOT = "" # project root


if not os.path.isdir(PROJECT_ROOT):
    raise NotADirectoryError(
    f"The specified PROJECT_ROOT path is either invalid or not a directory: {PROJECT_ROOT}\n"
    "Please check the path in your code."
)


BASE_PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "preprocessed_data")
BASE_LOG_PATH = os.path.join(PROJECT_ROOT, "log")  
BASE_SAVE_PATH = os.path.join(PROJECT_ROOT, "save")
BASE_CPKL_PATH = os.path.join(PROJECT_ROOT, "cpkl_basic")

PATHS = {
    "preprocessed_data_base": os.path.join(BASE_PREPROCESSED_DATA_PATH, "eth" + os.sep), #FOR ETH SCENARIO
    "log_base_directory": os.path.join(BASE_LOG_PATH, "basic", "eth" + os.sep), #FOR ETH SCENARIO
    "save_base_directory": os.path.join(BASE_SAVE_PATH, "basic", "eth" + os.sep), #FOR ETH SCENARIO
    "cpkl_directory": BASE_CPKL_PATH + os.sep,
    "trajectories_cpkl_filename": "trajectories.cpkl",
    "train_dataset_name": "att-train",
    "validation_dataset_name": "att-validation",
    "test_dataset_name": "att-test",
}

def get_trajectories_cpkl_path():
    return os.path.join(PATHS["cpkl_directory"], PATHS["trajectories_cpkl_filename"])

def get_data_dir(dataset_type_name):  
    return os.path.join(PATHS["preprocessed_data_base"], PATHS[dataset_type_name])

def get_log_directory(k_head_val):
    return os.path.join(PATHS["log_base_directory"], f'k={str(k_head_val)}/')

def get_save_directory(k_head_val):
    return os.path.join(PATHS["save_base_directory"], f'k={str(k_head_val)}/')

def get_checkpoint_path(save_dir, epoch):
    return os.path.join(save_dir, f'basic_lstm_model_{str(epoch)}.tar')

def get_trained_checkpoint_path(k_head_val, pretrained_model_index_val):
    save_dir = get_save_directory(k_head_val)
    model_filename = f'basic_lstm_model_{str(pretrained_model_index_val)}.tar' if pretrained_model_index_val is not None else 'best_model_statedict.pth'
    return os.path.join(save_dir, model_filename)

# Criterion

def mdn_loss(pi1, sigma1, mu1, data1, list_of_nodes):
    """
    Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG parameters.
    """
    if list_of_nodes.numel() == 0:
        return torch.tensor(0.0, device=pi1.device, requires_grad=True)

    pi = torch.index_select(pi1, 0, list_of_nodes)
    sigma = torch.index_select(sigma1, 0, list_of_nodes)
    mu = torch.index_select(mu1, 0, list_of_nodes)
    data = torch.index_select(data1, 0, list_of_nodes)

    prob = pi * mdn_gaussian_2d_likelihood(sigma, mu, data)

    epsilon = 1e-20
    nll = -torch.log(torch.clamp(torch.sum(prob, 1), min=epsilon))  
    return torch.mean(nll)


def mdn_sample(pi1, sigma1, mu1, list_of_nodes, state, device='cpu'):
    """
    Draw samples from a MoG during train/test
    state: {"train", "test"}
    """
    numNodes_total = pi1.size()[0]
    out_feature = sigma1.size()[2]
    sample1 = torch.zeros(numNodes_total, out_feature, device=device)
    
    if list_of_nodes.numel() == 0:
        return sample1, torch.empty(0, out_feature, device=device)

    list_of_nodes = list_of_nodes.to(pi1.device) 

    pi = torch.index_select(pi1, 0, list_of_nodes)
    sigma = torch.index_select(sigma1, 0, list_of_nodes)
    mu = torch.index_select(mu1, 0, list_of_nodes)

    if state == 'train':
        categorical = Categorical(pi)
        pis_indices = categorical.sample()
    else: # state == 'test'
        _, pis_indices = torch.max(pi, 1)

    idx_expanded = pis_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, mu.size(2))
    mu_selected = torch.gather(mu, 1, idx_expanded).squeeze(1)
    sigma_selected = torch.gather(sigma, 1, idx_expanded).squeeze(1)
    
    epsilon_sample = torch.randn_like(mu_selected)  
    sample_temp = mu_selected + sigma_selected * epsilon_sample
    
    if list_of_nodes.numel() > 0:
        sample1.index_add_(0, list_of_nodes, sample_temp)

    return sample1, sample_temp


def adefde(predict, targets, nodesPresent):
    """
    Calculate the ade/fde error
    predict: [NumAllNodes, 2]
    targets: [NumAllNodes, 2]
    nodesPresent: 1D Tensor of indices for present nodes
    """
    if nodesPresent.numel() == 0:
        return torch.tensor(0.0, device=predict.device)

    nodesPresent = nodesPresent.to(predict.device)

    predict_selected = torch.index_select(predict, 0, nodesPresent)
    targets_selected = torch.index_select(targets, 0, nodesPresent)

    diff = predict_selected - targets_selected
    if diff.numel() == 0:
        return torch.tensor(0.0, device=predict.device)

    ade = torch.sum(
        torch.sqrt(
            torch.sum(
                torch.mul(
                    diff,
                    diff),
                1)),
        0) / diff.size()[0]

    return ade


def mdn_gaussian_2d_likelihood(sigma, mu, target):
    """
    Calculates the likelihood of target data given 2D MoG parameters.
    sigma: [num_nodes, num_gaussians, 2] (std_x, std_y)
    mu: [num_nodes, num_gaussians, 2] (mean_x, mean_y)
    target: [num_nodes, 2] (target_x, target_y)
    """
    num_gaussians = sigma.size(1)
    num_nodes = sigma.size(0)
    
    target_expanded = target.unsqueeze(1).expand_as(mu)

    mux = mu[:, :, 0]
    muy = mu[:, :, 1]
    
    sx = torch.exp(sigma[:, :, 0]) + 1e-6
    sy = torch.exp(sigma[:, :, 1]) + 1e-6

    normx = target_expanded[:, :, 0] - mux
    normy = target_expanded[:, :, 1] - muy
    
    z = (normx / sx)**2 + (normy / sy)**2
    denom = 2 * np.pi * sx * sy
    denom = torch.clamp(denom, min=1e-20)

    result = torch.exp(-0.5 * z) / denom
    gaussian_prob = result

    return gaussian_prob

# Utility

def cal_curvature(traj):
    peds_traj = np.concatenate(np.array(traj, dtype=object)).astype(None)  
    peds_total = np.unique(peds_traj[:, 0]).tolist()
    overall_curvature_points = []

    for idx_ped in peds_total:
        a = peds_traj[peds_traj[:, 0] == idx_ped, 5:7].astype(float)  
        if len(a) <= 2:  
            continue
        
        dx_dt = np.gradient(a[:, 0])
        dy_dt = np.gradient(a[:, 1])
        
        ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

        tangent_x = np.zeros_like(dx_dt)
        tangent_y = np.zeros_like(dy_dt)
        non_zero_ds_dt_mask = ds_dt != 0
        if np.any(non_zero_ds_dt_mask):
            tangent_x[non_zero_ds_dt_mask] = dx_dt[non_zero_ds_dt_mask] / ds_dt[non_zero_ds_dt_mask]
            tangent_y[non_zero_ds_dt_mask] = dy_dt[non_zero_ds_dt_mask] / ds_dt[non_zero_ds_dt_mask]

        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)

        numerator = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)
        denominator = (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        
        curvature_values_ped = np.zeros_like(numerator)
        valid_denominator_mask = denominator != 0
        if np.any(valid_denominator_mask):
             curvature_values_ped[valid_denominator_mask] = numerator[valid_denominator_mask] / denominator[valid_denominator_mask]
        
        overall_curvature_points.extend(curvature_values_ped.tolist())

    return np.mean(overall_curvature_points) if overall_curvature_points else 0.0


def derivative(traj):
    peds_traj = np.concatenate(np.array(traj, dtype=object)).astype(None)  
    peds_total = np.unique(peds_traj[:, 0]).tolist()
    max_d_overall = 0.0

    for idx_ped in peds_total:
        a = peds_traj[peds_traj[:, 0] == idx_ped, 5:7].astype(float)  

        if len(a) < 2:  
            continue

        p2 = np.array([a[0, 0], a[0, 1]])
        p1 = np.array([a[-1, 0], a[-1, 1]])
        
        current_max_d_ped = 0.0
        if len(a) > 2:  
            for idx_point in range(1, len(a) - 1):  
                p3 = np.array([a[idx_point, 0], a[idx_point, 1]])
                
                vec_segment = p1 - p2  
                vec_point_to_start = p3 - p2

                cross_product_mag = np.abs(np.cross(vec_segment, vec_point_to_start))
                
                d2_sq = np.sum(vec_segment**2)  
                
                if d2_sq == 0:  
                    d = np.sqrt(np.sum((p3 - p2)**2))  
                else:
                    d2 = np.sqrt(d2_sq)
                    d = cross_product_mag / d2
                
                if d > current_max_d_ped:
                    current_max_d_ped = d
            
        if current_max_d_ped > max_d_overall:
            max_d_overall = current_max_d_ped
            
    return max_d_overall


# Data Structure & loader Classes 

class ST_NODE:
    def __init__(self, node_type, node_id, node_pos_list):
        self.node_type = node_type
        self.node_id = node_id
        # node_pos_list: dict: {framenum: (pedID, offx, offy, vx, vy, ax, ay, x, y, StopFlag)}
        self.node_pos_list = node_pos_list  

    def getPosition(self, index):
        assert(index in self.node_pos_list)
        return self.node_pos_list[index]

    def getType(self):
        return self.node_type

    def getID(self):
        return self.node_id

    def addPosition(self, pos, index):
        self.node_pos_list[index] = pos

    def printNode(self):
        print(f'Node type: {self.node_type} with ID: {self.node_id} '
              f'with positions: {list(self.node_pos_list.values())} '
              f'at time-steps: {list(self.node_pos_list.keys())}')


class ST_EDGE:
    def __init__(self, edge_type, edge_id, edge_pos_list):
        self.edge_type = edge_type
        self.edge_id = edge_id 
        self.edge_pos_list = edge_pos_list 

    def getPositions(self, index):
        assert(index in self.edge_pos_list)
        return self.edge_pos_list[index]

    def getType(self):
        return self.edge_type

    def getID(self):
        return self.edge_id

    def addPosition(self, pos, index):
        self.edge_pos_list[index] = pos

    def printEdge(self):
        print(f'Edge type: {self.edge_type} between nodes: {self.edge_id} '
              f'at time-steps: {list(self.edge_pos_list.keys())}')


class ST_GRAPH:
    def __init__(self, batch_size=50, seq_length=5): 
        self.initial_batch_size = batch_size
        self.batch_size = batch_size
        self.seq_length = seq_length  
        self.nodes = [{} for _ in range(self.batch_size)]
        self.edges = [{} for _ in range(self.batch_size)]

    def reset(self):
        self.nodes = [{} for _ in range(self.batch_size)]
        self.edges = [{} for _ in range(self.batch_size)]


    def readGraph(self, source_batch):
        if len(source_batch) != self.batch_size:
             self.batch_size = len(source_batch)
             self.nodes = [{} for _ in range(self.batch_size)]
             self.edges = [{} for _ in range(self.batch_size)]


        for sequence_idx in range(self.batch_size):
            source_seq = source_batch[sequence_idx]  
            
            if not isinstance(source_seq, (list, tuple)) or not all(isinstance(f, np.ndarray) for f in source_seq if f is not None):
                continue

            for framenum in range(min(len(source_seq), self.seq_length)):
                frame = source_seq[framenum]  

                if not isinstance(frame, np.ndarray) or frame.ndim != 2 or frame.shape[0] == 0:
                    continue

                for ped_row_idx in range(frame.shape[0]):
                    ped_data = frame[ped_row_idx, :]
                    pedID = ped_data[0]
                    # Expected: (PedID, offx, offy, vx, vy, ax, ay, x, y, StopFlag)
                    # Number of features is now 10 (or more if padded)
                    pos = tuple(ped_data[:10]) 

                    if pedID not in self.nodes[sequence_idx]:
                        self.nodes[sequence_idx][pedID] = ST_NODE('H', pedID, {framenum: pos})
                    else:
                        self.nodes[sequence_idx][pedID].addPosition(pos, framenum)

                        # Temporal edge
                        edge_id = (pedID, pedID)  
                        pos_edge_temporal = (pos, pos)  
                        
                        if edge_id not in self.edges[sequence_idx]:
                            self.edges[sequence_idx][edge_id] = ST_EDGE('H-H/T', edge_id, {framenum: pos_edge_temporal})
                        else:
                            self.edges[sequence_idx][edge_id].addPosition(pos_edge_temporal, framenum)
                
                # Add spatial edges
                current_peds_in_frame = frame[:, 0]  
                for i in range(len(current_peds_in_frame)):
                    for j in range(i + 1, len(current_peds_in_frame)):
                        pedID_in = current_peds_in_frame[i]
                        pedID_out = current_peds_in_frame[j]
                        
                        
                        pos_in_xy = (frame[i, 7], frame[i, 8]) 
                        pos_out_xy = (frame[j, 7], frame[j, 8])
                        pos_edge_spatial = (pos_in_xy, pos_out_xy)

                        edge_id_spatial = tuple(sorted((pedID_in, pedID_out)))  

                        if edge_id_spatial not in self.edges[sequence_idx]:
                            self.edges[sequence_idx][edge_id_spatial] = ST_EDGE('H-H/S', edge_id_spatial, {framenum: pos_edge_spatial})
                        else:
                            self.edges[sequence_idx][edge_id_spatial].addPosition(pos_edge_spatial, framenum)


    def printGraph(self):
        for sequence_idx in range(self.batch_size):
            nodes = self.nodes[sequence_idx]
            edges = self.edges[sequence_idx]
            print(f"\n--- Sequence {sequence_idx} ---")
            print('Printing Nodes')
            print('===============================')
            for node in nodes.values():
                node.printNode()
                print('--------------')
            print('Printing Edges')
            print('===============================')
            for edge in edges.values():
                edge.printEdge()
                print('--------------')

    def getSequence(self, ind):
        if ind >= self.batch_size or ind < 0:
            return np.zeros((self.seq_length, 0, 10)), \
                   np.zeros((self.seq_length, 0, 3)), \
                   [[] for _ in range(self.seq_length)], \
                   [[] for _ in range(self.seq_length)]


        nodes_dict = self.nodes[ind]  
        edges_dict = self.edges[ind]  

        pedID_to_local_idx = {ped_id: i for i, ped_id in enumerate(nodes_dict.keys())}
        numUniqueNodes = len(pedID_to_local_idx)

        # Updated to 10 features: (pedID, offx, offy, vx, vy, ax, ay, x, y, StopFlag)
        retNodes = np.zeros((self.seq_length, numUniqueNodes, 10))  
        retEdges = np.zeros((self.seq_length, numUniqueNodes * numUniqueNodes, 3))  
        
        retNodePresent = [[] for _ in range(self.seq_length)]
        retEdgePresent = [[] for _ in range(self.seq_length)]


        for original_pedID, node_instance in nodes_dict.items():
            if original_pedID not in pedID_to_local_idx: continue
            local_idx = pedID_to_local_idx[original_pedID]
            for framenum in range(self.seq_length):
                if framenum in node_instance.node_pos_list:
                    retNodePresent[framenum].append(local_idx)
                    node_pos_data = node_instance.node_pos_list[framenum] # This is a 10-element tuple
                    retNodes[framenum, local_idx, :] = list(node_pos_data)  
        
        for edge_tuple, edge_instance in edges_dict.items():
            ped_id1, ped_id2 = edge_tuple
            if ped_id1 not in pedID_to_local_idx or ped_id2 not in pedID_to_local_idx:
                continue

            local_idx1 = pedID_to_local_idx[ped_id1]
            local_idx2 = pedID_to_local_idx[ped_id2]

            for framenum in range(self.seq_length):
                if framenum in edge_instance.edge_pos_list:
                    if edge_instance.getType() == 'H-H/T':  
                        retEdgePresent[framenum].append((local_idx1, local_idx1))  
                    elif edge_instance.getType() == 'H-H/S':  
                        retEdgePresent[framenum].append((local_idx1, local_idx2))
                        retEdgePresent[framenum].append((local_idx2, local_idx1))  
        
        return retNodes, retEdges, retNodePresent, retEdgePresent


    def getBatch(self):
        return [self.getSequence(ind) for ind in range(self.batch_size)]


class TrainDataLoader:
    def __init__(
            self,
            datasets_names,
            seq_length,
            pred_length,
            batch_size,
            data_root_dir,
            infer=True,
            dt=0.4, 
            epsilon_stop_flag_val=0.1, 
            k_stop_flag_val=1.5
            ):
        
        self.datasets_names = datasets_names  
        self.data_root_dir = data_root_dir  
        
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.total_seq_length = seq_length + pred_length
        self.batch_size = batch_size
        self.infer = infer
        self.dt = dt
        self.epsilon_stop_flag_val = epsilon_stop_flag_val
        self.k_stop_flag_val = k_stop_flag_val

        self.data_file_cpkl = get_trajectories_cpkl_path()  

        self.idx_frame = 0  
        self.idx_d = 0      
        self.idx_batch = 0  
        
        self.dataset_file_index = 0  

        self.all_datasets_data = []  
        self.all_framelist_data = []
        self.all_numpeds_data = []
        self.all_frame_indices_data = []  
        self.trajectory_repository = {} 

        self.frame_preprocess()

        
        if self.infer and self.all_datasets_data:
            self.calculate_epsilon_stop_flag()


        self.double_data = []  
        self.dd_batch = 0      
        if self.infer:  
            self.cal_double_data()
            self.cal_double_batch_count()
            print(f"Number of double_data batches (high derivative): {self.dd_batch}")


    def calculate_epsilon_stop_flag(self):
        all_speeds = []
        for dataset_frames in self.all_datasets_data:
            for frame_data_list in dataset_frames:
                # frame_data_list is a list of np arrays for peds in a frame
                if isinstance(frame_data_list, np.ndarray) and frame_data_list.shape[0] > 0:
                    vx_values = frame_data_list[:, 3]
                    vy_values = frame_data_list[:, 4]
                    speeds = np.sqrt(vx_values**2 + vy_values**2)
                    all_speeds.extend(speeds.tolist())
        
        if all_speeds:
            median_speed = np.median(all_speeds)
            q1 = np.percentile(all_speeds, 25)
            q3 = np.percentile(all_speeds, 75)
            iqr = q3 - q1
            # Using IQR-based robust thresholding
            self.epsilon_stop_flag_val = max(0.01, iqr * self.k_stop_flag_val - median_speed) 
            print(f"Calculated epsilon_stop_flag_val for Stop Flag: {self.epsilon_stop_flag_val:.4f} (Median: {median_speed:.4f}, IQR: {iqr:.4f})")
        else:
            print("No speed data to calculate epsilon for Stop Flag. Using default.")


    def frame_preprocess(self):
        print(f"INFO (DataLoader): Starting frame_preprocess for {'training' if self.infer else 'validation'}...")
        print(f"INFO (DataLoader): data_root_dir: {self.data_root_dir}")
        print(f"INFO (DataLoader): datasets_names (files to process): {self.datasets_names}")
        print(f"INFO (DataLoader): Expected total sequence length (obs+pred): {self.total_seq_length}")
        
        if not self.datasets_names:
            print("WARNING (DataLoader): No dataset_names (files) provided to DataLoader. No data will be loaded.")
        
        for dataset_file_idx, dataset_file_name in enumerate(self.datasets_names):
            file_path = os.path.join(self.data_root_dir, dataset_file_name)
            print(f"INFO (DataLoader): Processing file [{dataset_file_idx+1}/{len(self.datasets_names)}]: {file_path}")
            
            current_file_frames_data = []  
            current_file_framelist = []
            current_file_numpeds = []
            current_file_frame_indices = []
            
            try:
                if not os.path.isfile(file_path):
                    print(f"ERROR (DataLoader): File not found at path: {file_path}. Skipping.")
                    self.all_datasets_data.append([]); self.all_framelist_data.append([])
                    self.all_numpeds_data.append([]); self.all_frame_indices_data.append([])
                    continue

                raw_data_from_file = np.genfromtxt(file_path, delimiter=',')
                if raw_data_from_file.ndim == 0 or raw_data_from_file.size == 0:  
                    print(f"WARNING (DataLoader): File {file_path} is empty. Skipping.")
                    self.all_datasets_data.append([]); self.all_framelist_data.append([])
                    self.all_numpeds_data.append([]); self.all_frame_indices_data.append([])
                    continue
                
               
                if raw_data_from_file.ndim == 2 and raw_data_from_file.shape[0] == 8 and raw_data_from_file.shape[1] > 8:
                    data_to_process = raw_data_from_file.T  
                elif raw_data_from_file.ndim == 2 and raw_data_from_file.shape[1] == 8:
                    data_to_process = raw_data_from_file
                elif raw_data_from_file.ndim == 1 and raw_data_from_file.shape[0] % 8 == 0 and raw_data_from_file.shape[0] > 0:
                    data_to_process = raw_data_from_file.reshape(-1, 8)
                else:
                    print(f"ERROR (DataLoader): Data shape {raw_data_from_file.shape} in {file_path} does not match expected formats. Skipping file.")
                    self.all_datasets_data.append([]); self.all_framelist_data.append([]); self.all_numpeds_data.append([]); self.all_frame_indices_data.append([])
                    continue

                if data_to_process is None or data_to_process.shape[0] == 0 or data_to_process.shape[1] < 8:
                    print(f"WARNING (DataLoader): No valid data to process for file {file_path}. Skipping.")
                    self.all_datasets_data.append([]); self.all_framelist_data.append([]); self.all_numpeds_data.append([]); self.all_frame_indices_data.append([])
                    continue
                
                # For AX, AY, StopFlag calculation
                ped_prev_data = defaultdict(lambda: {'pos': None, 'vel': None})
                
                unique_frames = np.unique(data_to_process[:, 0]).tolist()
                current_file_framelist = unique_frames

               
                if self.infer:
                    current_file_ped_trajectories = defaultdict(list)

                for frame_id in unique_frames:
                    peds_in_frame_observations = data_to_process[data_to_process[:, 0] == frame_id]
                    ped_ids_in_frame = np.unique(peds_in_frame_observations[:, 1]).tolist()
                    current_file_numpeds.append(len(ped_ids_in_frame))
                    
                    frame_peds_features = []
                    
                    if len(ped_ids_in_frame) > 0:
                        current_file_frame_indices.append(frame_id)  

                    for ped_id in ped_ids_in_frame:
                        ped_data_for_current_frame_ped = peds_in_frame_observations[peds_in_frame_observations[:, 1] == ped_id]
                        
                        if ped_data_for_current_frame_ped.shape[0] > 0:
                            entry = ped_data_for_current_frame_ped[0, :]

                            current_x, current_y = entry[6], entry[7] 
                            current_vx, current_vy = entry[4], entry[5] 
                            current_offx, current_offy = entry[2], entry[3] 

                            ax, ay = 0.0, 0.0
                            stop_flag = 0.0

                            if ped_prev_data[ped_id]['vel'] is not None:
                                prev_vx, prev_vy = ped_prev_data[ped_id]['vel']
                                # Compute ax, ay from velocity chang
                                ax = (current_vx - prev_vx) / self.dt
                                ay = (current_vy - prev_vy) / self.dt
                            
                            ped_prev_data[ped_id]['vel'] = (current_vx, current_vy)

                            # Stop Flag calculation
                            if abs(current_vx) < self.epsilon_stop_flag_val and abs(current_vy) < self.epsilon_stop_flag_val:
                                stop_flag = 1.0

                            # New 10 features: (PedID, offx, offy, vx, vy, ax, ay, x, y, StopFlag)
                            ped_features = [
                                ped_id,              # 0: PedID
                                current_offx,        # 1: offx
                                current_offy,        # 2: offy
                                current_vx,          # 3: vx
                                current_vy,          # 4: vy
                                ax,                  # 5: ax
                                ay,                  # 6: ay
                                current_x,           # 7: x
                                current_y,           # 8: y
                                stop_flag            # 9: StopFlag
                            ]
                            frame_peds_features.append(ped_features)

                            if self.infer:
                                current_file_ped_trajectories[ped_id].append([current_x, current_y])

                    if frame_peds_features:
                        frame_peds_np = np.array(frame_peds_features, dtype=np.float32)
                        padded_frame_peds_np = np.pad(frame_peds_np, ((0, 0), (0, 3)), 'constant', constant_values=0)
                        current_file_frames_data.append(padded_frame_peds_np)
                    elif len(ped_ids_in_frame) > 0:
                        current_file_frames_data.append(np.zeros((0, 13), dtype=np.float32)) 
                
                self.all_datasets_data.append(current_file_frames_data)
                self.all_framelist_data.append(current_file_framelist)
                self.all_numpeds_data.append(current_file_numpeds)
                self.all_frame_indices_data.append(current_file_frame_indices)

                
                if self.infer:
                    for ped_id, traj_points in current_file_ped_trajectories.items():
                        if len(traj_points) >= self.total_seq_length:
                            self.trajectory_repository[f"{dataset_file_name}_{ped_id}"] = np.array(traj_points)


            except Exception as e:
                print(f"ERROR (DataLoader): Exception while processing file {file_path}: {e}")
                import traceback
                traceback.print_exc()
                self.all_datasets_data.append([])
                self.all_framelist_data.append([])
                self.all_numpeds_data.append([])
                self.all_frame_indices_data.append([])
                continue
        
        counter = 0
        for i in range(len(self.all_datasets_data)):  
            num_frames_with_peds_in_file = len(self.all_datasets_data[i])  
            filename_for_log = self.datasets_names[i] if i < len(self.datasets_names) else f"Unknown File {i}"
            print(f"  INFO (DataLoader): File '{filename_for_log}' has {num_frames_with_peds_in_file} processed frames (with peds) in its list.")
            
            if num_frames_with_peds_in_file >= self.total_seq_length:
                sequences_from_file = (num_frames_with_peds_in_file - self.total_seq_length + 1)
                print(f"    -> Contributing {sequences_from_file} sequences to counter.")
                counter += sequences_from_file
            else:
                print(f"    -> Not enough frames with peds ({num_frames_with_peds_in_file}) for a sequence of length {self.total_seq_length}. Contributing 0 sequences.")
        
        print(f"INFO (DataLoader): Total sequences (counter) before dividing by batch_size: {counter}")
        print(f"INFO (DataLoader): Batch size: {self.batch_size}")
        self.num_batches = int(counter / self.batch_size) if self.batch_size > 0 else 0
        
        if self.all_datasets_data and self.dataset_file_index < len(self.all_datasets_data) and len(self.all_datasets_data[self.dataset_file_index]) > 0 :
            self.current_dataset_frames = self.all_datasets_data[self.dataset_file_index]
        else:
            self.current_dataset_frames = []
        
        print(f"INFO (DataLoader): Total number of {'training' if self.infer else 'validation'} batches calculated: {self.num_batches}")

    def cal_double_data(self):
        self.double_data = []  
        for i in range(len(self.all_datasets_data)):
            self.double_data.append([])
            current_frames = self.all_datasets_data[i]
            idx = 0
            while idx + self.total_seq_length <= len(current_frames):
                seq_source_frame_data = current_frames[idx : idx + self.total_seq_length]
                
                valid_sequence = True
                for frame_in_seq in seq_source_frame_data:
                    if not isinstance(frame_in_seq, np.ndarray) or frame_in_seq.shape[0] == 0:
                        valid_sequence = False
                        break
                if not valid_sequence:
                    idx += random.randint(1, self.seq_length) if self.seq_length > 0 else 1
                    continue

                curr_derivative_val = derivative(seq_source_frame_data)
                if curr_derivative_val > 0.70:
                    self.double_data[i].append(seq_source_frame_data)
                idx += random.randint(1, self.seq_length) if self.seq_length > 0 else 1

    def cal_double_batch_count(self):
        self.dd_batch = 0
        for i in range(len(self.double_data)):
            self.dd_batch += len(self.double_data[i])
        self.dd_batch = int(self.dd_batch / self.batch_size) if self.batch_size > 0 else 0


    def next_batch(self):
        s_data = []  
        m_data = []
        t_data = []

        current_batch_type_is_double = False
        if self.infer and self.dd_batch > 0 and \
           (self.num_batches == 0 or self.idx_batch >= self.num_batches):
            current_batch_type_is_double = True
            current_dd_batch_idx = self.idx_batch - self.num_batches if self.num_batches > 0 else self.idx_batch
            if current_dd_batch_idx >= self.dd_batch:
                return [], [], [], []


        filled_sequences = 0
        while filled_sequences < self.batch_size:
            if current_batch_type_is_double:
                if not self.double_data or self.dataset_file_index >= len(self.double_data) or not self.double_data[self.dataset_file_index]:
                    self.reset_double_dataset_pointer(forward=True)  
                    if self.dataset_file_index == 0 and self.idx_d == 0: break
                    continue

                current_file_double_data = self.double_data[self.dataset_file_index]
                if self.idx_d < len(current_file_double_data):
                    seq_to_add = current_file_double_data[self.idx_d]
                    s_data.append(seq_to_add)
                    m_data.append(None)  
                    filled_sequences += 1
                    self.idx_d += 1
                else:  
                    self.idx_d = 0
                    self.reset_double_dataset_pointer(forward=True)
                    if self.dataset_file_index == 0: break  
            else: 
                if not self.current_dataset_frames or self.dataset_file_index >= len(self.all_datasets_data) or not self.all_datasets_data[self.dataset_file_index]:
                    self.reset_dataset_pointer(forward=True)
                    if self.dataset_file_index == 0 and self.idx_frame == 0 : break
                    continue

                if self.idx_frame + self.total_seq_length <= len(self.current_dataset_frames):
                    seq_to_add = self.current_dataset_frames[self.idx_frame : self.idx_frame + self.total_seq_length]
                    s_data.append(seq_to_add)
                    m_data.append(None)  
                    filled_sequences += 1
                    
                    step_size = random.randint(1, self.seq_length) if self.infer and self.seq_length > 0 else 1
                    self.idx_frame += step_size
                else:  
                    self.idx_frame = 0
                    self.reset_dataset_pointer(forward=True)  
                    if self.dataset_file_index == 0 : break  
        
        if filled_sequences > 0:
            self.idx_batch += 1
        
        return s_data, m_data, t_data, []


    def reset_batch_pointer(self):
        self.idx_batch = 0
        self.idx_frame = 0  
        self.idx_d = 0      
        self.dataset_file_index = 0  
        if self.all_datasets_data and len(self.all_datasets_data) > 0:
             self.current_dataset_frames = self.all_datasets_data[0] if len(self.all_datasets_data[0]) > 0 else []
        else:
            self.current_dataset_frames = []


    def reset_dataset_pointer(self, forward=True):  
        if forward:
            self.dataset_file_index += 1
            if self.dataset_file_index >= len(self.all_datasets_data):
                self.dataset_file_index = 0  
        else:
            self.dataset_file_index -= 1
            if self.dataset_file_index < 0:
                self.dataset_file_index = len(self.all_datasets_data) - 1 if self.all_datasets_data else 0
        
        if self.all_datasets_data and self.dataset_file_index < len(self.all_datasets_data):  
            self.current_dataset_frames = self.all_datasets_data[self.dataset_file_index]
        else:  
            self.current_dataset_frames = []


    def reset_double_dataset_pointer(self, forward=True):  
        if forward:
            self.dataset_file_index += 1
            if not self.double_data or self.dataset_file_index >= len(self.double_data):  
                self.dataset_file_index = 0


    def next_sample_batch(self):
        s_data = []
        t_data = []
        if_end = False

        if not hasattr(self, 'idx_sample_frame_eval'): self.idx_sample_frame_eval = 0
        if not hasattr(self, 'dataset_file_index_eval'): self.dataset_file_index_eval = 0


        while len(s_data) < self.batch_size: 
            if self.dataset_file_index_eval >= len(self.all_datasets_data):
                if_end = True; break  

            current_frames = self.all_datasets_data[self.dataset_file_index_eval]
            
            if self.idx_sample_frame_eval + self.total_seq_length <= len(current_frames):
                obs_part = current_frames[self.idx_sample_frame_eval : self.idx_sample_frame_eval + self.seq_length]
                gt_full_seq = current_frames[self.idx_sample_frame_eval : self.idx_sample_frame_eval + self.total_seq_length]
                
                s_data.append(obs_part)
                t_data.append(gt_full_seq)
                
                self.idx_sample_frame_eval += 1
            else:  
                self.idx_sample_frame_eval = 0  
                self.dataset_file_index_eval += 1  
                if self.dataset_file_index_eval >= len(self.all_datasets_data):
                    if_end = True; break  
        
        return s_data, [], t_data, [], if_end
    
    def reset_sample_batch_pointer(self):
        self.idx_sample_frame_eval = 0
        self.dataset_file_index_eval = 0


class TestDataLoader:
    def __init__(
            self,
            datasets_names,
            seq_length,
            pred_length,
            batch_size,
            data_root_dir,
            dt=0.4,
            epsilon_stop_flag_val=0.1 
            ):
        
        self.datasets_names = datasets_names
        self.data_root_dir = data_root_dir
        
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.total_seq_length = seq_length + pred_length
        self.batch_size = batch_size
        self.dt = dt
        self.epsilon_stop_flag_val = epsilon_stop_flag_val 

        self.idx_frame = 0
        self.idx_batch = 0
        self.dataset_file_index = 0

        self.all_datasets_data = []
        self.all_framelist_data = []
        self.all_numpeds_data = []
        self.all_frame_indices_data = []  
        self.all_str_streams_data = []    

        self.frame_preprocess()
        
        if self.all_datasets_data and len(self.all_datasets_data) > 0 and len(self.all_datasets_data[self.dataset_file_index]) > 0:
            self.current_dataset_frames = self.all_datasets_data[self.dataset_file_index]
            self.current_dataset_strdata = self.all_str_streams_data[self.dataset_file_index] if self.all_str_streams_data and len(self.all_str_streams_data) > self.dataset_file_index else []
        else:
            self.current_dataset_frames = []
            self.current_dataset_strdata = []


    def frame_preprocess(self):
        for dataset_file_name in self.datasets_names:
            file_path = os.path.join(self.data_root_dir, dataset_file_name)
            try:
                raw_data = np.genfromtxt(file_path, delimiter=',')
                if raw_data.ndim == 0 or raw_data.size == 0:
                    print(f"Warning: Test file {file_path} is empty. Skipping.")
                    self.all_datasets_data.append([]); self.all_framelist_data.append([])
                    self.all_numpeds_data.append([]); self.all_frame_indices_data.append([])
                    self.all_str_streams_data.append([])
                    continue
            except Exception as e:
                print(f"Error reading test file {file_path}: {e}. Skipping.")
                self.all_datasets_data.append([]); self.all_framelist_data.append([])
                self.all_numpeds_data.append([]); self.all_frame_indices_data.append([])
                self.all_str_streams_data.append([])
                continue

            current_file_frames_data = []
            current_file_framelist = []
            current_file_numpeds = []
            current_file_frame_indices = []
            current_file_str_streams_per_frame = []

            if raw_data.ndim == 1:  
                if raw_data.shape[0] % 8 == 0 and raw_data.shape[0] > 0:
                    raw_data = raw_data.reshape(-1, 8)
                else:
                    print(f"Warning: Test file {file_path} is 1D and not a multiple of 8 features. Skipping.")
                    self.all_datasets_data.append([]); self.all_framelist_data.append([])
                    self.all_numpeds_data.append([]); self.all_frame_indices_data.append([])
                    self.all_str_streams_data.append([])
                    continue
            
            if raw_data.shape[1] < 8:
                print(f"Warning: Test file {file_path} has less than 8 columns ({raw_data.shape[1]}). Skipping.")
                self.all_datasets_data.append([]); self.all_framelist_data.append([])
                self.all_numpeds_data.append([]); self.all_frame_indices_data.append([])
                self.all_str_streams_data.append([])
                continue

            ped_prev_data = defaultdict(lambda: {'pos': None, 'vel': None}) # For AX, AY, StopFlag

            unique_frames = np.unique(raw_data[:, 0]).tolist()
            current_file_framelist = unique_frames

            for frame_id in unique_frames:
                peds_in_frame_data = raw_data[raw_data[:, 0] == frame_id]
                ped_ids_in_frame = np.unique(peds_in_frame_data[:, 1]).tolist()
                current_file_numpeds.append(len(ped_ids_in_frame))
                
                frame_peds_features = []
                frame_str_streams_temp = []

                if len(ped_ids_in_frame) > 0:
                    current_file_frame_indices.append(frame_id)

                for ped_id in ped_ids_in_frame:
                    ped_data_in_frame = peds_in_frame_data[peds_in_frame_data[:, 1] == ped_id]
                    if ped_data_in_frame.shape[0] > 0:
                        entry = ped_data_in_frame[0, :]
                        
                        current_x, current_y = entry[6], entry[7]
                        current_vx, current_vy = entry[4], entry[5]
                        current_offx, current_offy = entry[2], entry[3]

                        ax, ay = 0.0, 0.0
                        stop_flag = 0.0

                        if ped_prev_data[ped_id]['vel'] is not None:
                            prev_vx, prev_vy = ped_prev_data[ped_id]['vel']
                            ax = (current_vx - prev_vx) / self.dt
                            ay = (current_vy - prev_vy) / self.dt
                        
                        ped_prev_data[ped_id]['vel'] = (current_vx, current_vy)

                        if abs(current_vx) < self.epsilon_stop_flag_val and abs(current_vy) < self.epsilon_stop_flag_val:
                            stop_flag = 1.0

                        # 10 features: (PedID, offx, offy, vx, vy, ax, ay, x, y, StopFlag)
                        ped_features = [
                            ped_id, current_offx, current_offy, current_vx, current_vy,
                            ax, ay, current_x, current_y, stop_flag
                        ]
                        frame_peds_features.append(ped_features)
                        
                        str_data_source = dataset_file_name.split('_')[0]  
                        frame_str_streams_temp.append([str_data_source])  

                if frame_peds_features:
                    frame_peds_np = np.array(frame_peds_features, dtype=np.float32)
                    padded_frame_peds_np = np.pad(frame_peds_np, ((0, 0), (0, 3)), 'constant', constant_values=0) 
                    current_file_frames_data.append(padded_frame_peds_np)
                    current_file_str_streams_per_frame.append(frame_str_streams_temp)
                elif len(ped_ids_in_frame) > 0:  
                    current_file_frames_data.append(np.zeros((0,13), dtype=np.float32))
                    current_file_str_streams_per_frame.append([])


            self.all_datasets_data.append(current_file_frames_data)
            self.all_framelist_data.append(current_file_framelist)
            self.all_numpeds_data.append(current_file_numpeds)
            self.all_frame_indices_data.append(current_file_frame_indices)
            self.all_str_streams_data.append(current_file_str_streams_per_frame)

        counter = 0
        for i in range(len(self.all_datasets_data)):
            num_frames_in_file = len(self.all_datasets_data[i])
            if num_frames_in_file >= self.total_seq_length:
                counter += max(0, num_frames_in_file - self.total_seq_length + 1)

        self.num_batches = int(np.ceil(counter / self.batch_size)) if self.batch_size > 0 else 0
        print(f'Total number of testing batches: {self.num_batches}')


    def next_batch(self):
        s_data_batch = []
        m_data_batch = []  

        filled_sequences_in_batch = 0
        while filled_sequences_in_batch < self.batch_size:
            if self.dataset_file_index >= len(self.all_datasets_data):
                break

            if not self.all_datasets_data[self.dataset_file_index]:
                self.idx_frame = 0
                self.dataset_file_index +=1
                if self.dataset_file_index < len(self.all_datasets_data):
                    self.current_dataset_frames = self.all_datasets_data[self.dataset_file_index]
                    self.current_dataset_strdata = self.all_str_streams_data[self.dataset_file_index] if self.all_str_streams_data else []
                else:
                    break
                continue


            current_file_data = self.all_datasets_data[self.dataset_file_index]
            current_file_str_streams = self.all_str_streams_data[self.dataset_file_index] if self.all_str_streams_data and len(self.all_str_streams_data) > self.dataset_file_index else [[] for _ in current_file_data]


            if self.idx_frame + self.total_seq_length <= len(current_file_data):
                seq_data = current_file_data[self.idx_frame : self.idx_frame + self.total_seq_length]
                seq_mask_data = current_file_str_streams[self.idx_frame : self.idx_frame + self.total_seq_length] if current_file_str_streams and len(current_file_str_streams) > self.idx_frame else []
                
                s_data_batch.append(seq_data)
                m_data_batch.append(seq_mask_data)  
                
                self.idx_frame += 1  
                filled_sequences_in_batch += 1
            else:
                self.idx_frame = 0
                self.dataset_file_index += 1
                if self.dataset_file_index < len(self.all_datasets_data):
                    self.current_dataset_frames = self.all_datasets_data[self.dataset_file_index]
                    self.current_dataset_strdata = self.all_str_streams_data[self.dataset_file_index] if self.all_str_streams_data else []
                else:
                    break
        
        if filled_sequences_in_batch > 0:
            self.idx_batch += 1
            
        return s_data_batch, m_data_batch, [], []


    def reset_batch_pointer(self):
        self.idx_batch = 0
        self.idx_frame = 0
        self.dataset_file_index = 0
        if self.all_datasets_data:
            self.current_dataset_frames = self.all_datasets_data[0]
            self.current_dataset_strdata = self.all_str_streams_data[0] if self.all_str_streams_data and len(self.all_str_streams_data) > 0 else []
        else:
            self.current_dataset_frames = []
            self.current_dataset_strdata = []

# model
#some of the line is for future works for such as GEM

class GoalEstimationModule(nn.Module):
    def __init__(self, args, trajectory_repository):
        super(GoalEstimationModule, self).__init__()
        self.args = args
        self.trajectory_repository = trajectory_repository # dict of {ped_id: np.array([x,y]...)}
        self.soft_dtw = SoftDTW(gamma=0.1, normalize=True) 
        self.kmeans_n_clusters = args.kmeans_clusters
        self.device = args.device 

    def forward(self, observed_trajectory_batch):

        if observed_trajectory_batch.numel() == 0:
             return torch.empty(0, 2, device=self.device) 

        initial_goals_xy = observed_trajectory_batch[:, -1, :] 
        return initial_goals_xy


class SocialInteractionModule(nn.Module):
    def __init__(self, args):
        super(SocialInteractionModule, self).__init__()
        self.args = args
        self.k_head = args.k_head
        self.rnn_size = args.rnn_size
        self.device = args.device 

        # DBSCAN parameters for Pre-Grouping
        
        self.dbscan_eps = args.dbscan_eps 
        self.dbscan_min_samples = args.dbscan_min_samples 

        # Interaction Embedding (phi_q)
       
        self.phi_q = nn.Sequential(
            nn.Linear(7, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, self.rnn_size)
        ).to(self.device)

        # Weight calculation for aggregation (phi_w, phi_g)
        self.phi_w = nn.Sequential(
            nn.Linear(self.rnn_size, 1),
            nn.ReLU() 
        ).to(self.device)

        self.phi_g = nn.Sequential( # For group-level 
            nn.Linear(self.rnn_size, 1),
            nn.ReLU()
        ).to(self.device)

        # Mode Classification (phi_c)
        self.num_interaction_modes = args.num_interaction_modes 
        self.phi_c = nn.Sequential(
            nn.Linear(self.rnn_size, self.num_interaction_modes),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        # Trainable weights (W1, W2)
        self.W1_agg = nn.Parameter(torch.rand(1, self.rnn_size))
        self.W2_agg = nn.Parameter(torch.rand(1, self.rnn_size))


    def _rela_transform_with_fov(self, nodes_data, list_of_present_nodes_indices, neighbor_size, FoV_angle):
        # nodes_data: (numNodes, 10_features) -> (PedID, offx, offy, vx, vy, ax, ay, x, y, StopFlag)
        # FoV_angle: Field of View angle in degrees

        numNodes = nodes_data.size(0)
        # Initialize visibility grid to zeros
        vepo_grid = torch.zeros(numNodes, numNodes, 1, device=self.device, dtype=torch.bool)

        if list_of_present_nodes_indices.numel() <= 1:
            return vepo_grid 


        coords = nodes_data[:, 7:9].float() 
        velocities = nodes_data[:, 3:5].float() 

        for i_idx_val in list_of_present_nodes_indices:
            i = i_idx_val.item()
            agent_pos = coords[i]
            agent_vel = velocities[i]

            for j_idx_val in list_of_present_nodes_indices:
                j = j_idx_val.item()
                if i == j: 
                    continue
                
                other_pos = coords[j]
                
                # Check distance
                dist_sq = torch.sum((agent_pos - other_pos)**2)
                if dist_sq > (neighbor_size**2):
                    continue # Not within distance threshold

                # Calculate relative position vector r_ij
                r_ij = other_pos - agent_pos # (xj - xi, yj - yi)


                vel_magnitude = torch.norm(agent_vel)
                r_ij_magnitude = torch.norm(r_ij)

                if vel_magnitude < 1e-6 or r_ij_magnitude < 1e-6: # Very small movement or no relative position

                    if vel_magnitude < 1e-6 and dist_sq < (neighbor_size**2): # if agent is stopped but close, consider in FoV
                        vepo_grid[i, j, 0] = True
                    continue  

                dot_product = torch.dot(r_ij, agent_vel)
                
                cosine_theta_ij = torch.clamp(dot_product / (r_ij_magnitude * vel_magnitude), -1.0 + 1e-7, 1.0 - 1e-7)
                theta_ij = torch.acos(cosine_theta_ij) # Angle in radians

                # Check FoV condition
                if theta_ij <= torch.deg2rad(torch.tensor(FoV_angle / 2.0, device=self.device)):
                    vepo_grid[i, j, 0] = True
        
        return vepo_grid # Return as boolean tensor to be used as mask


    def forward(self, current_frame_nodes_data, h_nodes_prev, list_of_present_nodes_indices, args):

        numAllNodes = current_frame_nodes_data.size(0)
        # C: Final aggregated social tensor (numAllNodes, rnn_size)
        C = torch.zeros(numAllNodes, self.rnn_size, device=self.device)

        if list_of_present_nodes_indices.numel() == 0:
            return C

        # --- Step 1: Pre-Grouping Phase (DBSCAN) ---
        # Features for DBSCAN: [x, y, vx, vy, angle_to_center]
        
        relevant_features_for_clustering = current_frame_nodes_data[:, 3:9].float()
        
        nodes_to_cluster_temp = relevant_features_for_clustering[list_of_present_nodes_indices, :]
        node_ids_in_cluster_order = current_frame_nodes_data[list_of_present_nodes_indices, 0].detach().cpu().numpy() # PedIDs

        if nodes_to_cluster_temp.numel() == 0:
            return C

        # Normalize features for DBSCAN for better performance
        
        # Using (x,y,vx,vy) for clustering.

        column_indices_for_clustering = torch.tensor([7, 8, 3, 4], dtype=torch.long, device=self.device)
        
        data_for_present_nodes = torch.index_select(current_frame_nodes_data, 0, list_of_present_nodes_indices)
        
        clustering_features_tensor = torch.index_select(data_for_present_nodes, 1, column_indices_for_clustering)
        
        clustering_features = clustering_features_tensor.float().detach().cpu().numpy() # x,y,vx,vy
        
        
        if clustering_features.shape[0] > 1: 
            scaler = StandardScaler()
            scaled_clustering_features = scaler.fit_transform(clustering_features)
        else:
            scaled_clustering_features = clustering_features 

        # DBSCAN for grouping

        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        clusters = dbscan.fit_predict(scaled_clustering_features) # labels: -1 for noise, 0, 1, ... for clusters

        groups = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            if cluster_id != -1: # Ignore noise points
                original_global_idx = list_of_present_nodes_indices[i].item()
                groups[cluster_id].append(original_global_idx)
        
        # --- Step 2 & 3: Individual Mode Extraction & Group Embedding Phase ---
        # Et_ij: Interaction Embedding for each pair (i,j) in a group
        # Et_Gk: Group Embedding for each group Gk

        group_embeddings = {} # {cluster_id: Et_Gk (tensor, rnn_size)}

        for cluster_id, member_indices in groups.items():
            if len(member_indices) < 2: 
                continue
            
            # Calculate pairwise interaction embeddings for members of this group
            interaction_embeddings_for_group = [] # List of Et_ij
            for i_ped_idx in member_indices:
                for j_ped_idx in member_indices:
                    if i_ped_idx == j_ped_idx: continue

                    # Features for phi_q: [Δx, Δy, Δvx, Δvy, Δax, Δay, θ] -> 7 features
                    # current_frame_nodes_data: (PedID, offx, offy, vx, vy, ax, ay, x, y, StopFlag)
                    
                    # Absolute positions (x,y)
                    pos_i = current_frame_nodes_data[i_ped_idx, 7:9].float()
                    pos_j = current_frame_nodes_data[j_ped_idx, 7:9].float()
                    # Velocities (vx,vy)
                    vel_i = current_frame_nodes_data[i_ped_idx, 3:5].float()
                    vel_j = current_frame_nodes_data[j_ped_idx, 3:5].float()
                    # Accelerations (ax,ay)
                    acc_i = current_frame_nodes_data[i_ped_idx, 5:7].float()
                    acc_j = current_frame_nodes_data[j_ped_idx, 5:7].float()

                    # Calculate relative features
                    delta_pos = pos_j - pos_i # Δx, Δy
                    delta_vel = vel_j - vel_i # Δvx, Δvy
                    delta_acc = acc_j - acc_i # Δax, Δay

                    # Calculate angle theta_ij (between relative position and agent's velocity)
                    r_ij_vec = delta_pos
                    v_i_vec = vel_i
                    
                    r_ij_magnitude = torch.norm(r_ij_vec)
                    v_i_magnitude = torch.norm(v_i_vec)

                    theta_ij = torch.tensor(0.0, device=self.device) 
                    if v_i_magnitude > 1e-6 and r_ij_magnitude > 1e-6:
                        dot_product = torch.dot(r_ij_vec, v_i_vec)
                        cosine_theta_ij = torch.clamp(dot_product / (r_ij_magnitude * v_i_magnitude), -1.0 + 1e-7, 1.0 - 1e-7)
                        theta_ij = torch.acos(cosine_theta_ij) 

                    interaction_features = torch.cat([delta_pos, delta_vel, delta_acc, theta_ij.unsqueeze(0)], dim=0)
                    
                    Et_ij = self.phi_q(interaction_features) 
                    interaction_embeddings_for_group.append(Et_ij)
            
            if interaction_embeddings_for_group:
                # Group-level embedding (mean aggregation for simplicity, could be weighted)
                Et_Gk = torch.mean(torch.stack(interaction_embeddings_for_group), dim=0)
                group_embeddings[cluster_id] = Et_Gk


        # --- Step 4 & 5: Mode Classification & Group Labeling Phase ---
        # For each group

        group_mode_vectors = {} 
        for cluster_id, Et_Gk in group_embeddings.items():
            zt_Gk = self.phi_c(Et_Gk) 
            group_mode_vectors[cluster_id] = zt_Gk
        

        # --- Step 6: Mode Aggregation - Final Aggregation Phase (C) ---
        # calculate C_i (individual-level) and C_Gk (group-level)
        # Then combine them to get the final C.

        # Individual-leve (C_i): for each agent
        
        # Calculate visibility grid (FoV-based neighbors)
        visibility_grid_fov = self._rela_transform_with_fov(
            current_frame_nodes_data, list_of_present_nodes_indices, args.neighbor_size, args.fov_angle
        )

        Ci_per_node = torch.zeros(numAllNodes, self.rnn_size, device=self.device)

        for i_idx_val in list_of_present_nodes_indices:
            i = i_idx_val.item()
            neighbors_in_fov_indices = torch.where(visibility_grid_fov[i, :, 0])[0]

            if neighbors_in_fov_indices.numel() == 0:
                continue

            # Compute Et_ij for these neighbors

            individual_interaction_embeddings = []
            attention_weights_for_i = [] # For alpha_ij

            for j_idx_val in neighbors_in_fov_indices:
                j = j_idx_val.item()
                # Features for phi_q: [Δx, Δy, Δvx, Δvy, Δax, Δay, θ]
                pos_i = current_frame_nodes_data[i, 7:9].float()
                pos_j = current_frame_nodes_data[j, 7:9].float()
                vel_i = current_frame_nodes_data[i, 3:5].float()
                vel_j = current_frame_nodes_data[j, 3:5].float()
                acc_i = current_frame_nodes_data[i, 5:7].float()
                acc_j = current_frame_nodes_data[j, 5:7].float()

                delta_pos = pos_j - pos_i
                delta_vel = vel_j - vel_i
                delta_acc = acc_j - acc_i

                r_ij_vec = delta_pos
                v_i_vec = vel_i
                r_ij_magnitude = torch.norm(r_ij_vec)
                v_i_magnitude = torch.norm(v_i_vec)
                theta_ij = torch.tensor(0.0, device=self.device)
                if v_i_magnitude > 1e-6 and r_ij_magnitude > 1e-6:
                    dot_product = torch.dot(r_ij_vec, v_i_vec)
                    cosine_theta_ij = torch.clamp(dot_product / (r_ij_magnitude * v_i_magnitude), -1.0 + 1e-7, 1.0 - 1e-7)
                    theta_ij = torch.acos(cosine_theta_ij)

                interaction_features = torch.cat([delta_pos, delta_vel, delta_acc, theta_ij.unsqueeze(0)], dim=0)
                Et_ij = self.phi_q(interaction_features)
                individual_interaction_embeddings.append(Et_ij)
                
                attention_weights_for_i.append(self.phi_w(Et_ij))

            if individual_interaction_embeddings:
                stacked_embeddings = torch.stack(individual_interaction_embeddings) 
                stacked_attention_scores = torch.stack(attention_weights_for_i)
                
                alpha_ij_softmax = torch.softmax(stacked_attention_scores.squeeze(-1), dim=0).unsqueeze(-1)
                
                Ci_per_node[i] = torch.sum(alpha_ij_softmax * stacked_embeddings, dim=0)
            
        # Group-level (CGk): Use already computed group_embeddings (Et_Gk)
        CGk_per_node = torch.zeros(numAllNodes, self.rnn_size, device=self.device)
        for i_idx_val in list_of_present_nodes_indices:
            i = i_idx_val.item()
            
            try:
                # Use numpy for efficient element
                idx_in_present_nodes = (list_of_present_nodes_indices == i_idx_val).nonzero(as_tuple=True)[0]
                if idx_in_present_nodes.numel() > 0:
                    assigned_cluster_id = clusters[idx_in_present_nodes.item()] # Get cluster ID using the local index in clustering_features
                else:
                    assigned_cluster_id = -1 
            except IndexError:
                assigned_cluster_id = -1


            if assigned_cluster_id != -1:
                if assigned_cluster_id in group_embeddings:
                    
                    # For alpha_Gk, it's defined as a softmax over groups.
                    all_group_ids = list(group_embeddings.keys())
                    if not all_group_ids:
                        continue

                    # Calculate attention weights for all groups for the current agent i
                    group_embeddings_list = [group_embeddings[gid] for gid in all_group_ids]
                    
                    # use the group embedding to compute its attention weight.
                    
                    # For current agent i, calculate attention weights over ALL groups
                    group_attention_scores = []
                    for Et_Gk_val in group_embeddings_list:
                        group_attention_scores.append(self.phi_g(Et_Gk_val))
                    
                    if group_attention_scores:
                        stacked_group_attention_scores = torch.stack(group_attention_scores).squeeze(-1) 
                        alpha_Gk_softmax = torch.softmax(stacked_group_attention_scores, dim=0).unsqueeze(-1) 

                        stacked_group_embeddings = torch.stack(group_embeddings_list) 
                        CGk_per_node[i] = torch.sum(alpha_Gk_softmax * stacked_group_embeddings, dim=0) 
        
        # Final combined aggregation
        C = self.W1_agg * Ci_per_node + self.W2_agg * CGk_per_node
        return C


class GoalRefinementModule(nn.Module):
    def __init__(self, args):
        super(GoalRefinementModule, self).__init__()
        self.args = args
        # GRM
        # input
        input_dim = 2 + args.rnn_size + 6  
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Output: [x_hat_goal, y_hat_goal]
        ).to(args.device)

    def forward(self, initial_goal_xy, social_tensor_C, current_state_st):


        if initial_goal_xy.numel() == 0 or social_tensor_C.numel() == 0 or current_state_st.numel() == 0:
            return initial_goal_xy 
        
        concatenated_input = torch.cat([initial_goal_xy, social_tensor_C, current_state_st], dim=-1) 
        
        refined_goal_xy = self.mlp(concatenated_input)
        return refined_goal_xy

# SocialLSTM

class Interp_SocialLSTM(nn.Module):
    def __init__(self, args, state):
        super(Interp_SocialLSTM, self).__init__()
        self.args = args
        self.num_gaussians = args.num_gaussians
        self.use_cuda = args.use_cuda
        self.device = args.device 
        self.seq_length = args.seq_length
        self.pred_length = args.pred_length
        self.total_seq_length = args.seq_length + args.pred_length
        self.state = state # "train" or "test"

        self.rnn_size = args.rnn_size
        self.input_size = args.input_size 
        self.output_size_mdn_dim = args.output_size # 2 for x,y offsets

        # Two LSTMCells for 2-layer LSTM
        self.lstm1 = nn.LSTMCell(self.rnn_size, self.rnn_size).to(self.device)
        self.lstm2 = nn.LSTMCell(self.rnn_size, self.rnn_size).to(self.device)
        
        # GRM input to LSTM is now [C, g_t', s_t]
        self.lstm_input_dim = self.rnn_size + 2 + 6
        self.embedding_lstm_input_layer = nn.Linear(self.lstm_input_dim, self.rnn_size).to(self.device)
        
        #  hidden state
        self.pi_layer = nn.Linear(self.rnn_size, self.num_gaussians).to(self.device)  
        self.sigma_layer = nn.Linear(self.rnn_size, self.num_gaussians * self.output_size_mdn_dim).to(self.device)  
        self.mu_layer = nn.Linear(self.rnn_size, self.num_gaussians * self.output_size_mdn_dim).to(self.device)  

        self.softmax_pi_gaussians = nn.Softmax(dim=1).to(self.device)

        self.social_module = SocialInteractionModule(args).to(self.device)
        self.goal_refinement_module = GoalRefinementModule(args).to(self.device)


    def train_one_step(self,  
                             current_frame_nodes_raw, 
                             nodes_present_in_frame_indices,  
                             h_prev1, c_prev1, h_prev2, c_prev2, # For 2-layer LSTM
                             initial_goals_for_peds, 
                             args): 
        numAllNodes = current_frame_nodes_raw.size(0)
        
        pi_all = torch.zeros(numAllNodes, self.num_gaussians, device=self.device)
        mu_all = torch.zeros(numAllNodes, self.num_gaussians, self.output_size_mdn_dim, device=self.device)
        sigma_all = torch.zeros(numAllNodes, self.num_gaussians, self.output_size_mdn_dim, device=self.device)
        
        h_new1 = h_prev1.clone(); c_new1 = c_prev1.clone()
        h_new2 = h_prev2.clone(); c_new2 = c_prev2.clone()

        if nodes_present_in_frame_indices.numel() == 0:
            return pi_all, mu_all, sigma_all, h_new1, c_new1, h_new2, c_new2

        h_current_peds1 = h_prev1[nodes_present_in_frame_indices]
        c_current_peds1 = c_prev1[nodes_present_in_frame_indices]
        h_current_peds2 = h_prev2[nodes_present_in_frame_indices]
        c_current_peds2 = c_prev2[nodes_present_in_frame_indices]

        current_state_st_all_nodes = current_frame_nodes_raw[:, [7,8,3,4,5,6]].float() 
        current_state_st_present_peds = current_state_st_all_nodes[nodes_present_in_frame_indices]

        initial_goals_present_peds = initial_goals_for_peds[nodes_present_in_frame_indices] 


        #Social Interaction Module
        social_tensor_C_all_nodes = self.social_module(
            current_frame_nodes_raw.float(), h_prev1, nodes_present_in_frame_indices, args)
        social_tensor_C_present_peds = social_tensor_C_all_nodes[nodes_present_in_frame_indices]


        #Goal Refinement Module (outputs g_t')
        # Input: (initial_goal_xy, social_tensor_C, current_state_st)
        refined_goals_present_peds = self.goal_refinement_module(
            initial_goals_present_peds, social_tensor_C_present_peds, current_state_st_present_peds
        ) 

        lstm_input_features_present_peds = torch.cat([
            social_tensor_C_present_peds,        
            refined_goals_present_peds,          
            current_state_st_present_peds        
        ], dim=-1) 
        
        embedded_lstm_input = self.embedding_lstm_input_layer(lstm_input_features_present_peds)


        # --- 2-layer LSTM ---
        h_updated_current_peds1, c_updated_current_peds1 = self.lstm1(embedded_lstm_input, (h_current_peds1, c_current_peds1))
        h_updated_current_peds2, c_updated_current_peds2 = self.lstm2(h_updated_current_peds1, (h_current_peds2, c_current_peds2)) # Input of LSTM2 is output of LSTM1

        # Use final layer's hidden state for MDN outputs

        h_final = h_updated_current_peds2

        pi_logits_current_peds = self.pi_layer(h_final)  
        pi_current_peds = self.softmax_pi_gaussians(pi_logits_current_peds)    
        
        mu_current_peds = self.mu_layer(h_final).view(-1, self.num_gaussians, self.output_size_mdn_dim)
        sigma_raw_current_peds = self.sigma_layer(h_final).view(-1, self.num_gaussians, self.output_size_mdn_dim)
        sigma_current_peds = torch.exp(sigma_raw_current_peds)
        
        pi_all[nodes_present_in_frame_indices] = pi_current_peds
        mu_all[nodes_present_in_frame_indices] = mu_current_peds
        sigma_all[nodes_present_in_frame_indices] = sigma_current_peds
        
        h_new1[nodes_present_in_frame_indices] = h_updated_current_peds1
        c_new1[nodes_present_in_frame_indices] = c_updated_current_peds1
        h_new2[nodes_present_in_frame_indices] = h_updated_current_peds2
        c_new2[nodes_present_in_frame_indices] = c_updated_current_peds2
        
        return pi_all, mu_all, sigma_all, h_new1, c_new1, h_new2, c_new2


    def run_train(self, nodes_data_from_stgraph, present_nodes_indices_per_frame, args, trajectory_repository_for_goals):
        obs_len = args.seq_length
        pred_len = args.pred_length
        total_seq_len = obs_len + pred_len

        nodes_data_torch = torch.from_numpy(nodes_data_from_stgraph).float().to(self.device)
        numUniqueNodes = nodes_data_torch.size(1)

        ret_nodes_pred_traj = torch.zeros(total_seq_len, numUniqueNodes, 5, device=self.device)
        
        ret_nodes_pred_traj[:obs_len, :, 0] = nodes_data_torch[:obs_len, :, 0] # PedID
        ret_nodes_pred_traj[:obs_len, :, 3] = nodes_data_torch[:obs_len, :, 7] # abs_x
        ret_nodes_pred_traj[:obs_len, :, 4] = nodes_data_torch[:obs_len, :, 8] # abs_y

        loss_total = 0.0
        ade_errors_steps = []
        
        if not present_nodes_indices_per_frame or \
           len(present_nodes_indices_per_frame) < total_seq_len or \
           not nodes_data_torch.shape[1] > 0 or \
           not present_nodes_indices_per_frame[0] or \
           not present_nodes_indices_per_frame[total_seq_len-1] or \
           len(present_nodes_indices_per_frame[0]) == 0 or \
           len(present_nodes_indices_per_frame[total_seq_len-1]) == 0 :
            return torch.tensor(0.0, device=self.device, requires_grad=True), \
                   torch.tensor(float('inf'), device=self.device), \
                   torch.tensor(float('inf'), device=self.device), \
                   [ret_nodes_pred_traj]


        nodes_present_at_start_idx = torch.tensor(present_nodes_indices_per_frame[0], dtype=torch.long, device=self.device)
        nodes_present_at_end_idx = torch.tensor(present_nodes_indices_per_frame[total_seq_len-1], dtype=torch.long, device=self.device)
        
        common_nodes_for_fde_ade_mask = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device)
        if nodes_present_at_start_idx.numel() > 0 and nodes_present_at_end_idx.numel() > 0:
            mask_start = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device)
            mask_end = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device)
            mask_start[nodes_present_at_start_idx] = True
            mask_end[nodes_present_at_end_idx] = True
            common_nodes_for_fde_ade_mask = mask_start & mask_end
        
        common_nodes_indices_for_fde_ade = torch.where(common_nodes_for_fde_ade_mask)[0]

        if common_nodes_indices_for_fde_ade.numel() == 0:
             return torch.tensor(0.0, device=self.device, requires_grad=True), \
                    torch.tensor(float('inf'), device=self.device), \
                    torch.tensor(float('inf'), device=self.device), \
                    [ret_nodes_pred_traj]

        h_state1 = torch.zeros(numUniqueNodes, args.rnn_size, device=self.device)
        c_state1 = torch.zeros(numUniqueNodes, args.rnn_size, device=self.device)
        h_state2 = torch.zeros(numUniqueNodes, args.rnn_size, device=self.device)
        c_state2 = torch.zeros(numUniqueNodes, args.rnn_size, device=self.device)
        
        last_predicted_offsets_for_input = torch.zeros(numUniqueNodes, self.output_size_mdn_dim, device=self.device)

        observed_traj_abs_xy = nodes_data_torch[:obs_len, :, 7:9] 
        
   
        initial_goals_for_all_peds = torch.zeros(numUniqueNodes, 2, device=self.device)
        
        
        peds_present_in_obs = set()
        for t_idx in range(obs_len):
            peds_present_in_obs.update(present_nodes_indices_per_frame[t_idx])
        
        
        peds_present_in_obs_tensor = torch.tensor(list(peds_present_in_obs), dtype=torch.long, device=self.device)

        if peds_present_in_obs_tensor.numel() > 0:

            
            last_obs_frame_idx = obs_len -1
            peds_present_last_obs_frame = torch.tensor(present_nodes_indices_per_frame[last_obs_frame_idx], dtype=torch.long, device=self.device)
            if peds_present_last_obs_frame.numel() > 0:
                initial_goals_for_all_peds[peds_present_last_obs_frame] = nodes_data_torch[last_obs_frame_idx, peds_present_last_obs_frame, 7:9]

        for frame_idx in range(obs_len):
            nodes_present_in_this_frame = torch.tensor(present_nodes_indices_per_frame[frame_idx], dtype=torch.long, device=self.device)
            
            # Train_one_step now needs nodes_data_torch[frame_idx] (full features), 
            # h_state1, c_state1, h_state2, c_state2, initial_goals_for_all_peds
            pi, mu, sigma, h_state1, c_state1, h_state2, c_state2 = self.train_one_step(
                nodes_data_torch[frame_idx],  
                nodes_present_in_this_frame,
                h_state1, c_state1, h_state2, c_state2,
                initial_goals_for_all_peds, 
                args 
            )

            if frame_idx < obs_len - 1:
                target_offsets_next_frame = nodes_data_torch[frame_idx + 1, :, 1:3] 
                nodes_present_in_next_frame = torch.tensor(present_nodes_indices_per_frame[frame_idx+1], dtype=torch.long, device=self.device)
                
                if nodes_present_in_this_frame.numel() > 0 and nodes_present_in_next_frame.numel() > 0:
                    mask_curr = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device)
                    mask_next = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device)
                    mask_curr[nodes_present_in_this_frame] = True
                    mask_next[nodes_present_in_next_frame] = True
                    common_nodes_for_loss_indices = torch.where(mask_curr & mask_next)[0]

                    if common_nodes_for_loss_indices.numel() > 0:
                        loss = mdn_loss(pi, sigma, mu, target_offsets_next_frame, common_nodes_for_loss_indices)
                        loss_total += loss
            
            if nodes_present_in_this_frame.numel() > 0:
                predicted_offsets, _ = mdn_sample(pi, sigma, mu, nodes_present_in_this_frame, self.state, self.device)
                last_predicted_offsets_for_input.zero_()
                last_predicted_offsets_for_input[nodes_present_in_this_frame] = predicted_offsets[nodes_present_in_this_frame]

                if frame_idx + 1 < total_seq_len:
                    ret_nodes_pred_traj[frame_idx + 1, nodes_present_in_this_frame, 1] = predicted_offsets[nodes_present_in_this_frame, 0] 
                    ret_nodes_pred_traj[frame_idx + 1, nodes_present_in_this_frame, 2] = predicted_offsets[nodes_present_in_this_frame, 1] 
                    ret_nodes_pred_traj[frame_idx + 1, nodes_present_in_this_frame, 3] = ret_nodes_pred_traj[frame_idx, nodes_present_in_this_frame, 3] + predicted_offsets[nodes_present_in_this_frame, 0]
                    ret_nodes_pred_traj[frame_idx + 1, nodes_present_in_this_frame, 4] = ret_nodes_pred_traj[frame_idx, nodes_present_in_this_frame, 4] + predicted_offsets[nodes_present_in_this_frame, 1]
                    ret_nodes_pred_traj[frame_idx + 1, :, 0] = nodes_data_torch[frame_idx+1,:,0] # PedID 

            if frame_idx < obs_len -1 and 'common_nodes_for_loss_indices' in locals() and common_nodes_for_loss_indices.numel() > 0:
                pred_abs_next_step = ret_nodes_pred_traj[frame_idx+1, :, 3:5]
                true_abs_next_step = nodes_data_torch[frame_idx+1, :, 7:9] # abs_x, abs_y 
                ade_err_step = adefde(pred_abs_next_step, true_abs_next_step, common_nodes_for_loss_indices)
                ade_errors_steps.append(ade_err_step)


        # Decoder
        for frame_idx in range(obs_len, total_seq_len -1):
            nodes_present_in_this_frame_pred = torch.tensor(present_nodes_indices_per_frame[frame_idx], dtype=torch.long, device=self.device)
            
            if nodes_present_in_this_frame_pred.numel() == 0:
                if frame_idx > 0 and frame_idx + 1 < total_seq_len :
                    ret_nodes_pred_traj[frame_idx + 1, :, 3:5] = ret_nodes_pred_traj[frame_idx, :, 3:5] 
                    ret_nodes_pred_traj[frame_idx + 1, :, 1:3] = 0 
                    ret_nodes_pred_traj[frame_idx + 1, :, 0] = ret_nodes_pred_traj[frame_idx, :, 0] 
                continue

            temp_nodes_raw_for_decoder_step = torch.zeros(numUniqueNodes, 13, device=self.device)
            temp_nodes_raw_for_decoder_step[:, 0] = ret_nodes_pred_traj[frame_idx, :, 0] 
            
            pred_offsets_curr_step = last_predicted_offsets_for_input[nodes_present_in_this_frame_pred]
            temp_nodes_raw_for_decoder_step[nodes_present_in_this_frame_pred, 1:3] = pred_offsets_curr_step 
            
            pred_vx_curr_step = pred_offsets_curr_step[:, 0] / args.dt
            pred_vy_curr_step = pred_offsets_curr_step[:, 1] / args.dt
            temp_nodes_raw_for_decoder_step[nodes_present_in_this_frame_pred, 3] = pred_vx_curr_step
            temp_nodes_raw_for_decoder_step[nodes_present_in_this_frame_pred, 4] = pred_vy_curr_step

            current_abs_pos_predicted = ret_nodes_pred_traj[frame_idx, :, 3:5].detach()
            temp_nodes_raw_for_decoder_step[:, 7:9] = current_abs_pos_predicted 
            temp_nodes_raw_for_decoder_step[:, 5:7] = 0.0 
            temp_nodes_raw_for_decoder_step[:, 9] = 0.0 

            pi, mu, sigma, h_state1, c_state1, h_state2, c_state2 = self.train_one_step(
                temp_nodes_raw_for_decoder_step,  
                nodes_present_in_this_frame_pred,
                h_state1, c_state1, h_state2, c_state2,
                initial_goals_for_all_peds, 
                args 
            )
            
            target_offsets_next_frame_gt = nodes_data_torch[frame_idx + 1, :, 1:3] 
            nodes_present_in_next_frame_gt = torch.tensor(present_nodes_indices_per_frame[frame_idx+1], dtype=torch.long, device=self.device)

            if nodes_present_in_this_frame_pred.numel() > 0 and nodes_present_in_next_frame_gt.numel() > 0:
                mask_curr_dec = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device)
                mask_next_dec = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device)
                mask_curr_dec[nodes_present_in_this_frame_pred] = True
                mask_next_dec[nodes_present_in_next_frame_gt] = True
                common_nodes_for_loss_dec_indices = torch.where(mask_curr_dec & mask_next_dec)[0]

                if common_nodes_for_loss_dec_indices.numel() > 0:
                    loss = mdn_loss(pi, sigma, mu, target_offsets_next_frame_gt, common_nodes_for_loss_dec_indices)
                    loss_total += loss
            
            if nodes_present_in_this_frame_pred.numel() > 0:
                predicted_offsets_step, _ = mdn_sample(pi, sigma, mu, nodes_present_in_this_frame_pred, self.state, self.device)
                last_predicted_offsets_for_input.zero_()
                last_predicted_offsets_for_input[nodes_present_in_this_frame_pred] = predicted_offsets_step[nodes_present_in_this_frame_pred]

                ret_nodes_pred_traj[frame_idx + 1, nodes_present_in_this_frame_pred, 1] = predicted_offsets_step[nodes_present_in_this_frame_pred, 0]
                ret_nodes_pred_traj[frame_idx + 1, nodes_present_in_this_frame_pred, 2] = predicted_offsets_step[nodes_present_in_this_frame_pred, 1]
                ret_nodes_pred_traj[frame_idx + 1, nodes_present_in_this_frame_pred, 3] = ret_nodes_pred_traj[frame_idx, nodes_present_in_this_frame_pred, 3] + predicted_offsets_step[nodes_present_in_this_frame_pred, 0]
                ret_nodes_pred_traj[frame_idx + 1, nodes_present_in_this_frame_pred, 4] = ret_nodes_pred_traj[frame_idx, nodes_present_in_this_frame_pred, 4] + predicted_offsets_step[nodes_present_in_this_frame_pred, 1]
                ret_nodes_pred_traj[frame_idx + 1, :, 0] = ret_nodes_pred_traj[frame_idx,:,0]

            if common_nodes_indices_for_fde_ade.numel() > 0 and nodes_present_in_next_frame_gt.numel() > 0:
                mask_common_metrics = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device); mask_common_metrics[common_nodes_indices_for_fde_ade] = True
                mask_next_gt = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device); mask_next_gt[nodes_present_in_next_frame_gt] = True
                final_common_nodes_for_ade_step = torch.where(mask_common_metrics & mask_next_gt)[0]

                if final_common_nodes_for_ade_step.numel() > 0:
                    pred_abs_next_decoder = ret_nodes_pred_traj[frame_idx+1, :, 3:5]
                    true_abs_next_decoder = nodes_data_torch[frame_idx+1, :, 7:9] 
                    ade_err_step = adefde(pred_abs_next_decoder, true_abs_next_decoder, final_common_nodes_for_ade_step)
                    ade_errors_steps.append(ade_err_step)

        mean_ade_traj = torch.mean(torch.stack(ade_errors_steps)) if ade_errors_steps else torch.tensor(float('inf'), device=self.device)
        
        fde_traj = torch.tensor(float('inf'), device=self.device)
        if common_nodes_indices_for_fde_ade.numel() > 0:
            fde_traj = adefde(ret_nodes_pred_traj[total_seq_len-1, :, 3:5],
                              nodes_data_torch[total_seq_len-1, :, 7:9],
                              common_nodes_indices_for_fde_ade)
            
        return loss_total, mean_ade_traj, fde_traj, [ret_nodes_pred_traj]


    def run_test(self, nodes_data_from_stgraph, present_nodes_indices_per_frame, args, num_samples=20):
        obs_len = args.seq_length
        pred_len = args.pred_length
        total_seq_len = obs_len + pred_len

        min_ade_overall = float('inf')
        min_fde_overall = float('inf')
        best_ret_nodes_list_for_min_ade = []

        nodes_data_torch = torch.from_numpy(nodes_data_from_stgraph).float().to(self.device)
        numUniqueNodes = nodes_data_torch.size(1)

        if not present_nodes_indices_per_frame or \
           len(present_nodes_indices_per_frame) < total_seq_len or \
           not nodes_data_torch.shape[1] > 0 or \
           not present_nodes_indices_per_frame[0] or \
           not present_nodes_indices_per_frame[total_seq_len-1] or \
           len(present_nodes_indices_per_frame[0]) == 0 or \
           len(present_nodes_indices_per_frame[total_seq_len-1]) == 0:
            return torch.tensor(0.0, device=self.device), \
                   torch.tensor(float('inf'), device=self.device), \
                   torch.tensor(float('inf'), device=self.device), []

        nodes_present_at_start_idx = torch.tensor(present_nodes_indices_per_frame[0], dtype=torch.long, device=self.device)
        nodes_present_at_end_idx = torch.tensor(present_nodes_indices_per_frame[total_seq_len-1], dtype=torch.long, device=self.device)
        
        common_nodes_for_metrics_mask = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device)
        if nodes_present_at_start_idx.numel() > 0 and nodes_present_at_end_idx.numel() > 0:
            mask_start = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device); mask_start[nodes_present_at_start_idx] = True
            mask_end = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device); mask_end[nodes_present_at_end_idx] = True
            common_nodes_for_metrics_mask = mask_start & mask_end
        common_nodes_indices_for_metrics = torch.where(common_nodes_for_metrics_mask)[0]

        if common_nodes_indices_for_metrics.numel() == 0:
            return torch.tensor(0.0, device=self.device), \
                   torch.tensor(float('inf'), device=self.device), \
                   torch.tensor(float('inf'), device=self.device), []

        initial_goals_for_all_peds = torch.zeros(numUniqueNodes, 2, device=self.device)
        last_obs_frame_idx = obs_len -1
        peds_present_last_obs_frame = torch.tensor(present_nodes_indices_per_frame[last_obs_frame_idx], dtype=torch.long, device=self.device)
        if peds_present_last_obs_frame.numel() > 0:
            initial_goals_for_all_peds[peds_present_last_obs_frame] = nodes_data_torch[last_obs_frame_idx, peds_present_last_obs_frame, 7:9]

        for sample_idx in range(num_samples):
            ret_nodes_pred_traj_sample = torch.zeros(total_seq_len, numUniqueNodes, 5, device=self.device)
            ret_nodes_pred_traj_sample[:obs_len, :, 0] = nodes_data_torch[:obs_len, :, 0]
            ret_nodes_pred_traj_sample[:obs_len, :, 3] = nodes_data_torch[:obs_len, :, 7]
            ret_nodes_pred_traj_sample[:obs_len, :, 4] = nodes_data_torch[:obs_len, :, 8]

            current_sample_ade_errors_steps = []
            h_state1_sample = torch.zeros(numUniqueNodes, args.rnn_size, device=self.device)
            c_state1_sample = torch.zeros(numUniqueNodes, args.rnn_size, device=self.device)
            h_state2_sample = torch.zeros(numUniqueNodes, args.rnn_size, device=self.device)
            c_state2_sample = torch.zeros(numUniqueNodes, args.rnn_size, device=self.device)
            last_predicted_offsets_for_input_sample = torch.zeros(numUniqueNodes, self.output_size_mdn_dim, device=self.device)


            for frame_idx in range(obs_len):
                nodes_present_this_step = torch.tensor(present_nodes_indices_per_frame[frame_idx], dtype=torch.long, device=self.device)
                
                pi, mu, sigma, h_state1_sample, c_state1_sample, h_state2_sample, c_state2_sample = self.train_one_step(
                    nodes_data_torch[frame_idx], nodes_present_this_step,
                    h_state1_sample, c_state1_sample, h_state2_sample, c_state2_sample,
                    initial_goals_for_all_peds,
                    args) 
                if nodes_present_this_step.numel() > 0:
                    predicted_offsets, _ = mdn_sample(pi, sigma, mu, nodes_present_this_step, self.state, self.device)
                    last_predicted_offsets_for_input_sample.zero_()
                    last_predicted_offsets_for_input_sample[nodes_present_this_step] = predicted_offsets[nodes_present_this_step]
                    
                    if frame_idx + 1 < total_seq_len :  
                        ret_nodes_pred_traj_sample[frame_idx + 1, nodes_present_this_step, 1] = predicted_offsets[nodes_present_this_step, 0]
                        ret_nodes_pred_traj_sample[frame_idx + 1, nodes_present_this_step, 2] = predicted_offsets[nodes_present_this_step, 1]
                        ret_nodes_pred_traj_sample[frame_idx + 1, nodes_present_this_step, 3] = ret_nodes_pred_traj_sample[frame_idx, nodes_present_this_step, 3] + predicted_offsets[nodes_present_this_step, 0]
                        ret_nodes_pred_traj_sample[frame_idx + 1, nodes_present_this_step, 4] = ret_nodes_pred_traj_sample[frame_idx, nodes_present_this_step, 4] + predicted_offsets[nodes_present_this_step, 1]
                        ret_nodes_pred_traj_sample[frame_idx + 1, :, 0] = nodes_data_torch[frame_idx+1,:,0] if frame_idx+1 < total_seq_len else nodes_data_torch[frame_idx,:,0]

            for frame_idx in range(obs_len, total_seq_len - 1):
                nodes_present_this_step_dec = torch.tensor(present_nodes_indices_per_frame[frame_idx], dtype=torch.long, device=self.device)
                if nodes_present_this_step_dec.numel() == 0:
                    if frame_idx > 0 and frame_idx + 1 < total_seq_len:
                        ret_nodes_pred_traj_sample[frame_idx + 1, :, 3:5] = ret_nodes_pred_traj_sample[frame_idx, :, 3:5]
                        ret_nodes_pred_traj_sample[frame_idx + 1, :, 1:3] = 0
                        ret_nodes_pred_traj_sample[frame_idx + 1, :, 0] = ret_nodes_pred_traj_sample[frame_idx, :, 0]
                    continue

               
                temp_nodes_raw_dec = torch.zeros(numUniqueNodes, 13, device=self.device)
                temp_nodes_raw_dec[:, 0] = ret_nodes_pred_traj_sample[frame_idx, :, 0]
                
                pred_offsets_curr_step_sample = last_predicted_offsets_for_input_sample[nodes_present_this_step_dec]
                temp_nodes_raw_dec[nodes_present_this_step_dec, 1:3] = pred_offsets_curr_step_sample
                
                pred_vx_curr_step_sample = pred_offsets_curr_step_sample[:, 0] / args.dt
                pred_vy_curr_step_sample = pred_offsets_curr_step_sample[:, 1] / args.dt
                temp_nodes_raw_dec[nodes_present_this_step_dec, 3] = pred_vx_curr_step_sample
                temp_nodes_raw_dec[nodes_present_this_step_dec, 4] = pred_vy_curr_step_sample

                current_abs_pos_sample = ret_nodes_pred_traj_sample[frame_idx, :, 3:5].detach()
                temp_nodes_raw_dec[:, 7:9] = current_abs_pos_sample

                temp_nodes_raw_dec[:, 5:7] = 0.0 # ax, ay 
                temp_nodes_raw_dec[:, 9] = 0.0 # StopFlag 

                pi, mu, sigma, h_state1_sample, c_state1_sample, h_state2_sample, c_state2_sample = self.train_one_step(
                    temp_nodes_raw_dec, nodes_present_this_step_dec,
                    h_state1_sample, c_state1_sample, h_state2_sample, c_state2_sample,
                    initial_goals_for_all_peds,
                    args) 
                
                if nodes_present_this_step_dec.numel() > 0:
                    predicted_offsets_step_sample, _ = mdn_sample(pi, sigma, mu, nodes_present_this_step_dec, self.state, self.device)
                    last_predicted_offsets_for_input_sample.zero_()
                    last_predicted_offsets_for_input_sample[nodes_present_this_step_dec] = predicted_offsets_step_sample[nodes_present_this_step_dec]

                    ret_nodes_pred_traj_sample[frame_idx + 1, nodes_present_this_step_dec, 1] = predicted_offsets_step_sample[nodes_present_this_step_dec, 0]
                    ret_nodes_pred_traj_sample[frame_idx + 1, nodes_present_this_step_dec, 2] = predicted_offsets_step_sample[nodes_present_this_step_dec, 1]
                    ret_nodes_pred_traj_sample[frame_idx + 1, nodes_present_this_step_dec, 3] = ret_nodes_pred_traj_sample[frame_idx, nodes_present_this_step_dec, 3] + predicted_offsets_step_sample[nodes_present_this_step_dec, 0]
                    ret_nodes_pred_traj_sample[frame_idx + 1, nodes_present_this_step_dec, 4] = ret_nodes_pred_traj_sample[frame_idx, nodes_present_this_step_dec, 4] + predicted_offsets_step_sample[nodes_present_this_step_dec, 1]
                    ret_nodes_pred_traj_sample[frame_idx + 1, :, 0] = ret_nodes_pred_traj_sample[frame_idx,:,0]

               
                nodes_present_in_next_frame_gt_step = torch.tensor(present_nodes_indices_per_frame[frame_idx+1], dtype=torch.long, device=self.device)
                if common_nodes_indices_for_metrics.numel() > 0 and nodes_present_in_next_frame_gt_step.numel() > 0:
                    mask_common_metrics = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device); mask_common_metrics[common_nodes_indices_for_metrics] = True
                    mask_next_gt = torch.zeros(numUniqueNodes, dtype=torch.bool, device=self.device); mask_next_gt[nodes_present_in_next_frame_gt_step] = True
                    final_common_nodes_for_ade_step = torch.where(mask_common_metrics & mask_next_gt)[0]

                    if final_common_nodes_for_ade_step.numel() > 0:
                        ade_err_sample_step = adefde(ret_nodes_pred_traj_sample[frame_idx+1, :, 3:5],
                                                     nodes_data_torch[frame_idx+1, :, 7:9],
                                                     final_common_nodes_for_ade_step)
                        current_sample_ade_errors_steps.append(ade_err_sample_step)

            # Calculate ADE and FDE
            current_sample_mean_ade = torch.mean(torch.stack(current_sample_ade_errors_steps)).item() if current_sample_ade_errors_steps else float('inf')
            
            current_sample_fde = float('inf')
            if common_nodes_indices_for_metrics.numel() > 0:
                        current_sample_fde = adefde(ret_nodes_pred_traj_sample[total_seq_len-1, :, 3:5],
                                                     nodes_data_torch[total_seq_len-1, :, 7:9],
                                                     common_nodes_indices_for_metrics).item()

            if current_sample_mean_ade < min_ade_overall:
                min_ade_overall = current_sample_mean_ade
                min_fde_overall = current_sample_fde
                best_ret_nodes_list_for_min_ade = [ret_nodes_pred_traj_sample.clone()]
            elif current_sample_mean_ade == min_ade_overall and current_sample_fde < min_fde_overall :
                min_fde_overall = current_sample_fde
                best_ret_nodes_list_for_min_ade = [ret_nodes_pred_traj_sample.clone()]


        return torch.tensor(0.0, device=self.device), \
               torch.tensor(min_ade_overall, device=self.device), \
               torch.tensor(min_fde_overall, device=self.device), \
               best_ret_nodes_list_for_min_ade

# Traning FUN.

def train_epoch(dataloader, net, args, stgraph, epoch, optimizer, log_file_curve):
    dataloader.reset_batch_pointer()
    loss_epoch = 0.0
    num_processed_batches = 0

    total_batches_for_epoch = dataloader.num_batches
    if dataloader.infer and dataloader.dd_batch > 0:
        total_batches_for_epoch += dataloader.dd_batch

    batch_iterator = tqdm.tqdm(range(total_batches_for_epoch), desc=f"Epoch {epoch+1} Training")

    # Verify model and a tensor are on GPU at the start of epoch
    if args.device.type == 'cuda':
        print(f"DEBUG: Model is on device: {next(net.parameters()).device}")
        dummy_tensor = torch.ones(1).to(args.device)
        print(f"DEBUG: Dummy tensor is on device: {dummy_tensor.device}")
        del dummy_tensor 

    for batch_idx in batch_iterator:
        start_time = time.time()
        s_data_batch, _, _, _ = dataloader.next_batch()

        if not s_data_batch :  
            if batch_idx >= total_batches_for_epoch and not dataloader.infer :
                    break
            continue

        current_batch_size = len(s_data_batch)
        if current_batch_size == 0:  
            batch_iterator.set_description(f"Epoch {epoch+1} Training (Skipped empty batch)")
            continue

        stgraph.batch_size = current_batch_size
        stgraph.reset()
        stgraph.readGraph(s_data_batch)

        loss_batch_sum = 0
        ade_batch_sum = 0
        fde_batch_sum = 0
        valid_sequences_in_batch = 0

        for sequence_in_batch_idx in range(current_batch_size):
            nodes_temp, _, nodesPresent, _ = stgraph.getSequence(sequence_in_batch_idx)

            if not nodesPresent or len(nodesPresent) < args.seq_length + args.pred_length or \
               not nodes_temp.shape[1] > 0 or \
               not nodesPresent[0] or \
               len(nodesPresent[0]) == 0 or \
               not nodesPresent[args.seq_length + args.pred_length -1] or \
               len(nodesPresent[args.seq_length + args.pred_length -1]) == 0:
                continue
            
            net.train()  
            optimizer.zero_grad()
            
            loss_seq, ade_seq, fde_seq, _ = net.run_train(nodes_temp, nodesPresent, args, dataloader.trajectory_repository)
            
            if isinstance(loss_seq, torch.Tensor) and loss_seq.item() > 0 and torch.isfinite(loss_seq) :  
                loss_seq.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()
                loss_batch_sum += loss_seq.item()
                if torch.isfinite(ade_seq): ade_batch_sum += ade_seq.item()
                if torch.isfinite(fde_seq): fde_batch_sum += fde_seq.item()
                valid_sequences_in_batch +=1
        
        if valid_sequences_in_batch > 0:
            avg_loss_batch = loss_batch_sum / valid_sequences_in_batch
            avg_ade_batch = ade_batch_sum / valid_sequences_in_batch
            avg_fde_batch = fde_batch_sum / valid_sequences_in_batch
            loss_epoch += avg_loss_batch
            num_processed_batches +=1
            end_time = time.time()
            batch_iterator.set_postfix({
                'loss': f'{avg_loss_batch:.3f}',
                'ade': f'{avg_ade_batch:.3f}',
                'fde': f'{avg_fde_batch:.3f}'
            })

        else:
            pass

    if num_processed_batches > 0:
        avg_loss_epoch = loss_epoch / num_processed_batches
        log_file_curve.write(f'{epoch+1},{avg_loss_epoch}\n')  
        log_file_curve.flush()
        print(f'Epoch {epoch+1} average train loss: {avg_loss_epoch:.3f}')
    else:
        print(f"Epoch {epoch+1} had no valid batches for training.")
        log_file_curve.write(f'{epoch+1},nan\n')
        log_file_curve.flush()


def validation_epoch(dataloader, net, args, stgraph, epoch, log_val_curve):
    dataloader.reset_batch_pointer()
    dataloader.reset_sample_batch_pointer()

    loss_epoch_val = 0.0
    ade_epoch_val = 0.0
    fde_epoch_val = 0.0
    num_processed_batches_val = 0
    
    net.eval()  

    total_val_batches = dataloader.num_batches

    val_batch_iterator = tqdm.tqdm(range(total_val_batches), desc=f"Epoch {epoch+1} Validation", leave=False)


    with torch.no_grad():  
        for batch_idx in val_batch_iterator:
            s_data_batch, _, _, _ = dataloader.next_batch()

            if not s_data_batch:  
                continue

            current_batch_size = len(s_data_batch)
            if current_batch_size == 0:  
                if 'val_batch_iterator' in locals() and hasattr(val_batch_iterator, 'set_description'):
                    val_batch_iterator.set_description(f"Epoch {epoch+1} Validation (Skipped empty batch)")
                continue

            stgraph.batch_size = current_batch_size  
            stgraph.reset()                    
            stgraph.readGraph(s_data_batch)

            loss_batch_sum_val = 0
            ade_batch_sum_val = 0
            fde_batch_sum_val = 0
            valid_sequences_in_batch_val = 0

            for sequence_in_batch_idx in range(current_batch_size):
                nodes_temp, _, nodesPresent, _ = stgraph.getSequence(sequence_in_batch_idx)
                if not nodesPresent or len(nodesPresent) < args.seq_length + args.pred_length or \
                   not nodes_temp.shape[1] > 0 or \
                   not nodesPresent[0] or \
                   len(nodesPresent[0]) == 0 or \
                   not nodesPresent[args.seq_length + args.pred_length -1] or \
                   len(nodesPresent[args.seq_length + args.pred_length -1]) == 0:
                    continue
                
                loss_seq_val, ade_seq_val, fde_seq_val, _ = net.run_train(nodes_temp, nodesPresent, args, dataloader.trajectory_repository)
                
                if isinstance(loss_seq_val, torch.Tensor) and torch.isfinite(loss_seq_val) and loss_seq_val.item() >= 0 :
                    loss_batch_sum_val += loss_seq_val.item()
                    if torch.isfinite(ade_seq_val): ade_batch_sum_val += ade_seq_val.item()
                    if torch.isfinite(fde_seq_val): fde_batch_sum_val += fde_seq_val.item()
                    valid_sequences_in_batch_val += 1
            
            if valid_sequences_in_batch_val > 0:
                avg_loss_batch_val = loss_batch_sum_val / valid_sequences_in_batch_val
                avg_ade_batch_val = ade_batch_sum_val / valid_sequences_in_batch_val
                avg_fde_batch_val = fde_batch_sum_val / valid_sequences_in_batch_val

                loss_epoch_val += avg_loss_batch_val
                ade_epoch_val += avg_ade_batch_val
                fde_epoch_val += avg_fde_batch_val
                num_processed_batches_val += 1
                
                val_batch_iterator.set_postfix({
                    'val_loss': f'{avg_loss_batch_val:.3f}',
                    'val_ade': f'{avg_ade_batch_val:.3f}',
                    'val_fde': f'{avg_fde_batch_val:.3f}'
                })

    if num_processed_batches_val > 0:
        final_avg_loss_val = loss_epoch_val / num_processed_batches_val
        final_avg_ade_val = ade_epoch_val / num_processed_batches_val
        final_avg_fde_val = fde_epoch_val / num_processed_batches_val
        log_val_curve.write(f'{epoch+1},{final_avg_loss_val},{final_avg_ade_val},{final_avg_fde_val}\n')
        log_val_curve.flush()
        print(f'Epoch {epoch+1} average validation loss: {final_avg_loss_val:.3f}, ADE: {final_avg_ade_val:.3f}, FDE: {final_avg_fde_val:.3f}')
        return final_avg_loss_val, final_avg_ade_val, final_avg_fde_val
    else:
        print(f"Epoch {epoch+1} had no valid batches for validation.")
        log_val_curve.write(f'{epoch+1},nan,nan,nan\n')
        log_val_curve.flush()
        return float('inf'), float('inf'), float('inf')


def train_main_logic(args):
    print("Starting Training Process...")
    
    device = args.device
    print(f"Using device: {device}")

    # Explicit check if the selected device is CUDA
    if device.type == 'cuda':
        print(f"INFO: PyTorch is set to use CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"INFO: Current CUDA memory allocated: {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB")
        print(f"INFO: Current CUDA memory cached: {torch.cuda.memory_reserved(device) / (1024**2):.2f} MB")
    else:
        print("WARNING: Training is running on CPU. If you intended to use GPU, please check your arguments and CUDA installation.")

    seed = 8;  
    if args.use_cuda: torch.cuda.manual_seed_all(seed)  
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_dir = get_log_directory(args.k_head)
    save_dir = get_save_directory(args.k_head)
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'config_train.pkl'), 'wb') as f: pickle.dump(args, f)
    log_file_curve = open(os.path.join(log_dir, 'log_train_curve.txt'), 'w+')
    log_val_curve = open(os.path.join(log_dir, 'log_val_curve.txt'), 'w+')
    log_file_curve.write("epoch,train_loss\n")  
    log_val_curve.write("epoch,val_loss,val_ade,val_fde\n")  

    train_data_path = get_data_dir("train_dataset_name")
    val_data_path = get_data_dir("validation_dataset_name")

    print(f"INFO: Current Working Directory (from train_main_logic): {os.getcwd()}")
    abs_train_path = os.path.abspath(train_data_path)
    abs_val_path = os.path.abspath(val_data_path)
    print(f"INFO: Absolute training data path being used: {abs_train_path}")
    print(f"INFO: Absolute validation data path being used: {abs_val_path}")

    if not os.path.isdir(abs_train_path):
        raise FileNotFoundError(f"Training data directory not found: {abs_train_path}")
    if not os.path.isdir(abs_val_path):
        raise FileNotFoundError(f"Validation data directory not found: {abs_val_path}")
    
    train_files = [f for f in os.listdir(abs_train_path) if os.path.isfile(os.path.join(abs_train_path, f)) and f.endswith('.csv')]
    val_files = [f for f in os.listdir(abs_val_path) if os.path.isfile(os.path.join(abs_val_path, f)) and f.endswith('.csv')]

    if not train_files: raise FileNotFoundError(f"No training files (.csv) found in {abs_train_path}")
    if not val_files: raise FileNotFoundError(f"No validation files (.csv) found in {abs_val_path}")

    train_loader = TrainDataLoader(train_files, args.seq_length, args.pred_length, args.batch_size, train_data_path, infer=True,
                                   dt=args.dt, epsilon_stop_flag_val=args.epsilon_stop_flag, k_stop_flag_val=args.k_stop_flag_val)
    val_loader = TrainDataLoader(val_files, args.seq_length, args.pred_length, args.batch_size, val_data_path, infer=False,
                                  dt=args.dt, epsilon_stop_flag_val=args.epsilon_stop_flag, k_stop_flag_val=args.k_stop_flag_val)
    
    val_loader.epsilon_stop_flag_val = train_loader.epsilon_stop_flag_val

    total_traj_length = args.seq_length + args.pred_length
    stgraph_train = ST_GRAPH(args.batch_size, total_traj_length)
    stgraph_val = ST_GRAPH(args.batch_size, total_traj_length)

    net = Interp_SocialLSTM(args, state='train').to(device)
    print(net)  
    print(f"INFO: Model parameters are on device: {next(net.parameters()).device}")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, eps=1e-3, amsgrad=True, weight_decay=args.lambda_param)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience_early_stop = args.early_stopping_patience  

    print('Training begin')
    for epoch in range(args.num_epochs):
        train_epoch(train_loader, net, args, stgraph_train, epoch, optimizer, log_file_curve)
        
        val_loss, val_ade, val_fde = validation_epoch(val_loader, net, args, stgraph_val, epoch, log_val_curve)
        
        scheduler.step(val_loss)  

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            print(f'New best validation loss: {best_val_loss:.4f}. Saving model for epoch {epoch+1}...')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'best_val_loss': best_val_loss,
                'val_ade_at_best': val_ade,
                'val_fde_at_best': val_fde,
                'epsilon_stop_flag': train_loader.epsilon_stop_flag_val 
            }, get_checkpoint_path(save_dir, epoch + 1))  
            torch.save(net.state_dict(), os.path.join(save_dir, 'best_model_statedict.pth'))
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve for {epochs_no_improve} epoch(s). Current best: {best_val_loss:.4f}')

        if epochs_no_improve >= patience_early_stop:
            print(f'Early stopping triggered after {patience_early_stop} epochs without improvement.')
            break
            
    log_file_curve.close()
    log_val_curve.close()
    print('Training finished.')

def test_epoch_logic(dataloader, net, args, stgraph):  
    dataloader.reset_batch_pointer()
    ade_total = 0.0
    fde_total = 0.0
    num_valid_sequences_for_metrics = 0
    num_samples_for_min_ade_fde = args.num_test_samples  

    net.eval()  

    with torch.no_grad():
        for batch_idx in tqdm.tqdm(range(dataloader.num_batches), desc="Testing Batches"):
            s_data_batch, m_data_batch, _, _ = dataloader.next_batch()

            if not s_data_batch: continue
            
            current_batch_size = len(s_data_batch)
            if current_batch_size == 0: continue

            stgraph.reset()
            stgraph.batch_size = current_batch_size  
            stgraph.readGraph(s_data_batch)  

            for seq_idx_in_batch in range(current_batch_size):
                nodes_temp, _, nodesPresent, _ = stgraph.getSequence(seq_idx_in_batch)
                
                if not nodesPresent or len(nodesPresent) < args.seq_length + args.pred_length or \
                   not nodes_temp.shape[1] > 0 or \
                   not nodesPresent[0] or \
                   len(nodesPresent[0]) == 0 or \
                   not nodesPresent[args.seq_length + args.pred_length -1] or \
                   len(nodesPresent[args.seq_length + args.pred_length -1]) == 0:
                    continue

                _, ade_seq_tensor, fde_seq_tensor, _ = net.run_test(nodes_temp, nodesPresent, args, num_samples_for_min_ade_fde)
                
                ade_seq = ade_seq_tensor.item()
                fde_seq = fde_seq_tensor.item()

                if ade_seq is not None and fde_seq is not None and \
                   ade_seq != float('inf') and fde_seq != float('inf') and \
                   ade_seq > 1e-9 and fde_seq > 1e-9:
                    ade_total += ade_seq
                    fde_total += fde_seq
                    num_valid_sequences_for_metrics += 1
        
    if num_valid_sequences_for_metrics > 0:
        avg_ade = ade_total / num_valid_sequences_for_metrics
        avg_fde = fde_total / num_valid_sequences_for_metrics
        print(f'Final Test ADE: {avg_ade:.4f}, Final Test FDE: {avg_fde:.4f} over {num_valid_sequences_for_metrics} sequences.')
    else:
        print('No valid sequences found for testing to calculate ADE/FDE.')
        avg_ade, avg_fde = float('nan'), float('nan')

    return avg_ade, avg_fde


def test_main_logic(args):
    print("Starting Testing Process...")
    
    device = args.device 
    print(f"Using device: {device}")

    # Explicit check if the selected device is CUDA
    if device.type == 'cuda':
        print(f"INFO: PyTorch is set to use CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"INFO: Current CUDA memory allocated: {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB")
        print(f"INFO: Current CUDA memory cached: {torch.cuda.memory_reserved(device) / (1024**2):.2f} MB")
    else:
        print("WARNING: Testing is running on CPU. If you intended to use GPU, please check your arguments and CUDA installation.")

    seed = 9  
    if args.use_cuda: torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_data_path = get_data_dir("test_dataset_name")

    abs_test_path = os.path.abspath(test_data_path)
    print(f"INFO: Absolute testing data path being used: {abs_test_path}")

    if not os.path.isdir(abs_test_path):
        raise FileNotFoundError(f"Testing data directory not found: {abs_test_path}")

    test_files = [f for f in os.listdir(abs_test_path) if os.path.isfile(os.path.join(abs_test_path, f)) and f.endswith('.csv')]
    if not test_files: raise FileNotFoundError(f"No testing files (.csv) found in {abs_test_path}")

    epsilon_stop_flag_from_model = args.epsilon_stop_flag 
    if args.pretrained_model_path:
        if os.path.exists(args.pretrained_model_path):
            checkpoint = torch.load(args.pretrained_model_path, map_location=device)
            if 'args' in checkpoint and hasattr(checkpoint['args'], 'epsilon_stop_flag'):
                epsilon_stop_flag_from_model = checkpoint['args'].epsilon_stop_flag
                print(f"Loaded epsilon_stop_flag from model checkpoint: {epsilon_stop_flag_from_model:.4f}")
            else:
                print("Warning: epsilon_stop_flag not found in model checkpoint args. Using default from CLI.")
        else:
            print(f"Warning: Pretrained model path '{args.pretrained_model_path}' not found. Cannot load epsilon_stop_flag from it.")
    
    test_loader = TestDataLoader(test_files, args.seq_length, args.pred_length, args.batch_size, test_data_path,
                                 dt=args.dt, epsilon_stop_flag_val=epsilon_stop_flag_from_model)

    total_traj_length = args.seq_length + args.pred_length
    stgraph_test = ST_GRAPH(args.batch_size, total_traj_length)  

    net = Interp_SocialLSTM(args, state='test').to(device)
    print(f"INFO: Model parameters are on device: {next(net.parameters()).device}") 


    if args.pretrained_model_index is None and args.pretrained_model_path is None:
        default_best_model_path = os.path.join(get_save_directory(args.k_head), 'best_model_statedict.pth')
        if os.path.isfile(default_best_model_path):
            print(f"INFO: No specific model provided, attempting to load default 'best_model_statedict.pth' for k_head={args.k_head}")
            args.pretrained_model_path = default_best_model_path
        else:
             raise ValueError("For testing, either --pretrained_model_index or --pretrained_model_path must be provided, or 'best_model_statedict.pth' must exist.")


    if args.pretrained_model_path:
        checkpoint_file = args.pretrained_model_path
    else:  
        checkpoint_file = get_trained_checkpoint_path(args.k_head, args.pretrained_model_index)

    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    print(f'--- Loading checkpoint from {checkpoint_file} ---')
    checkpoint = torch.load(checkpoint_file, map_location=device)  
    
    if 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
        if 'epoch' in checkpoint: print(f'- Loaded checkpoint from epoch {checkpoint["epoch"]} -')
        # Ensure test args align with training args for model architecture
        if 'args' in checkpoint:
            loaded_train_args = checkpoint['args']
            
            args.rnn_size = loaded_train_args.rnn_size
            args.input_size = loaded_train_args.input_size
            args.output_size = loaded_train_args.output_size
            args.num_gaussians = loaded_train_args.num_gaussians
            args.fov_angle = loaded_train_args.fov_angle
            args.dbscan_eps = loaded_train_args.dbscan_eps
            args.dbscan_min_samples = loaded_train_args.dbscan_min_samples
            args.kmeans_clusters = loaded_train_args.kmeans_clusters
            args.num_interaction_modes = loaded_train_args.num_interaction_modes
            print("INFO: Updated args from loaded model checkpoint for consistency.")
            
    else:
        net.load_state_dict(checkpoint)
        print('- Loaded checkpoint (assumed raw state_dict) -')
        
    print("Model loaded successfully.")

    test_ade, test_fde = test_epoch_logic(test_loader, net, args, stgraph_test)
    
    results_filename = f"test_results_k{args.k_head}"
    if args.pretrained_model_index is not None:
        results_filename += f"_model_epoch{args.pretrained_model_index}"
    elif args.pretrained_model_path is not None:
        model_name_part = os.path.splitext(os.path.basename(args.pretrained_model_path))[0]
        results_filename += f"_{model_name_part}"
    results_filename += ".txt"
    
    log_base_for_test_results = PATHS["log_base_directory"]
    if not os.path.exists(log_base_for_test_results): os.makedirs(log_base_for_test_results)

    results_file_path = os.path.join(log_base_for_test_results, results_filename)

    with open(results_file_path, 'w') as f:
        f.write(f"Test ADE: {test_ade}\n")
        f.write(f"Test FDE: {test_fde}\n")
        f.write(f"Tested model: {checkpoint_file}\n")
        f.write(f"Test Args: {vars(args)}\n")
    print(f"Test results saved to {results_file_path}")
    print('Testing finished.')


def main():
    parser = argparse.ArgumentParser(description="Comprehensive script for Trajectory Prediction Model (Interp_SocialLSTM)")
    
    parser.add_argument('--mode', type=str, default=None, choices=['train', 'test'],
                        help='Mode to run: "train" or "test". If not provided when run directly, defaults to "train".')
    
    parser.add_argument('--k_head', type=int, default=3)
    parser.add_argument('--input_size', type=int, default=10)
    parser.add_argument('--rnn_size', type=int, default=64) 
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--seq_length', type=int, default=5) 
    parser.add_argument('--pred_length', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--neighbor_size', type=float, default=4.0)
    parser.add_argument('--num_gaussians', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--force_cuda', action='store_true', default=False)
    parser.add_argument('--dt', type=float, default=0.4)
    parser.add_argument('--epsilon_stop_flag', type=float, default=0.1)
    parser.add_argument('--k_stop_flag_val', type=float, default=1.5)
    parser.add_argument('--fov_angle', type=float, default=180.0)
    parser.add_argument('--kmeans_clusters', type=int, default=20)
    parser.add_argument('--dbscan_eps', type=float, default=0.5)
    parser.add_argument('--dbscan_min_samples', type=int, default=2)
    parser.add_argument('--num_interaction_modes', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--lambda_param', type=float, default=0.0005)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--pretrained_model_index', type=int, default=None)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--num_test_samples', type=int, default=20)

    args = parser.parse_args()

    if args.force_cuda:
        if not args.use_cuda:
            raise ValueError("Error: --force_cuda is set, but --use_cuda is False. Please enable --use_cuda.")
        if not torch.cuda.is_available():
            raise RuntimeError("Error: --force_cuda is set, but CUDA is not available on this system.")
        device = torch.device("cuda") 
        print("Forcing GPU usage. CUDA device selected.")
    elif args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available and --use_cuda is enabled. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available or --use_cuda is not enabled. Using CPU.")

    print(f"Using device: {device}")
    args.device = device # Pass device to args for modules

    if args.mode is None:
        print("INFO: --mode argument not provided via command line.")
        default_mode = 'train'
        print(f"INFO: Defaulting to '{default_mode}' mode.")
        print("INFO: To run in a different mode or with specific parameters, please provide command line arguments.")
        args.mode = default_mode
        
        if args.mode == 'train':
            print("INFO: Ensure training data is available at paths specified in PATHS dictionary.")
        elif args.mode == 'test':
            print("INFO: For test mode, ensure a pretrained model is specified via --pretrained_model_path or --pretrained_model_index,")
            print("INFO: or that 'best_model_statedict.pth' exists in the default save directory.")

    if args.mode == 'train':
        train_main_logic(args)
    elif args.mode == 'test':
        test_main_logic(args)
    else:
        print(f"Error: Unknown mode '{args.mode}'. Please use 'train' or 'test'.")
        parser.print_help()


if __name__ == '__main__':
    main()
	
