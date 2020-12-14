from spektral.data import Dataset, Graph
import os
from dataprep import get_labels, read_bin_velodyne
from graph_gen import adjacency
import numpy as np

class PCGraph(Dataset):
    def __init__(self, base_dir, seq_no=None, seq_list=None, stop_idx=None, test=False, **kwargs):
        assert (seq_no is not None) or (seq_list is not None), 'Provide either a specific seq id or list of seqs'
        self.stop_idx = stop_idx
        self.seq = seq_no
        self.seq_list = seq_list
        self.base_dir = base_dir
        self.test = test
        super().__init__(**kwargs)

    def read(self):
        list_of_graphs = []
        if self.seq is not None:
            velo_dir = os.path.join(self.base_dir, self.seq, 'velodyne')
            label_dir = os.path.join(self.base_dir, self.seq, 'labels')
            velo_files = os.listdir(velo_dir)
            label_files = os.listdir(label_dir)
            for i in range(self.stop_idx):
                curr_velo_path = os.path.join(velo_dir, velo_files[i])
                curr_label_path = os.path.join(label_dir, label_files[i])
                x = read_bin_velodyne(curr_velo_path)[:, :3]
                y = get_labels(curr_label_path)
                print('     Graph Construction -- Seq {} | Scan {} -- complete.'.format(self.seq, i))
                a = adjacency(x)
                list_of_graphs.append(Graph(x=x, a=a, y=y))

        if self.seq_list is not None:
            for id in self.seq_list:
                velo_dir = os.path.join(self.base_dir, id, 'velodyne')
                velo_files = os.listdir(velo_dir)

                if self.stop_idx is not None:
                    iter_range = range(self.stop_idx)
                else:
                    iter_range = range(len(velo_files))
                if self.test:
                    for i in iter_range:
                        curr_velo_path = os.path.join(velo_dir, velo_files[i])
                        x = read_bin_velodyne(curr_velo_path)[:, :3]
                        print('     Graph Construction -- Seq {} | Scan {} -- complete.'.format(id, i))
                        a = adjacency(x)
                        list_of_graphs.append(Graph(x=x, a=a))
                else:
                    label_dir = os.path.join(self.base_dir, id, 'labels')
                    label_files = os.listdir(label_dir)
                    for i in iter_range:
                        curr_velo_path = os.path.join(velo_dir, velo_files[i])
                        curr_label_path = os.path.join(label_dir, label_files[i])
                        x = read_bin_velodyne(curr_velo_path)[:, :3]
                        y = get_labels(curr_label_path)
                        print('     Graph Construction -- Seq {} | Scan {} -- complete.'.format(id, i))
                        a = adjacency(x)
                        list_of_graphs.append(Graph(x=x, a=a, y=y))
        print('     Preprocessing..')
        return list_of_graphs

