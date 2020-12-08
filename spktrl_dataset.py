from spektral.data import Dataset, Graph
import os
from dataprep import get_labels, read_bin_velodyne
from graph_gen import adjacency

class PCGraph(Dataset):
    def __init__(self, base_dir, seq_no, stop_idx=None, **kwargs):
        self.stop_idx = stop_idx
        self.velo_dir = os.path.join(base_dir, seq_no, 'velodyne')
        self.label_dir = os.path.join(base_dir, seq_no, 'labels')
        self.velo_files = os.listdir(self.velo_dir)
        self.label_files = os.listdir(self.label_dir)
        super().__init__(**kwargs)

    def read(self):
        list_of_graphs = []
        for i in range(self.stop_idx):
            curr_velo_path = os.path.join(self.velo_dir, self.velo_files[i])
            curr_label_path = os.path.join(self.label_dir, self.label_files[i])
            x = read_bin_velodyne(curr_velo_path)[:, :3]
            y = get_labels(curr_label_path)
            a = adjacency(x)
            list_of_graphs.append(Graph(x=x, a=a, y=y))
        return list_of_graphs

