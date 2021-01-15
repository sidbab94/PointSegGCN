from os import listdir
from os.path import join
import numpy as np
import struct
from yaml import safe_load


def read_bin_velodyne(pc_path):
    '''
    Reads velodyne binary file and converts it to a numpy.nd.array
    :param pc_path: SemanticKITTI binary scan file path
    :return: point cloud array
    '''
    pc_list = []
    with open(pc_path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def load_label_kitti(label_path, remap_lut):
    '''
    Reads .label file and maps it to a sparse one-dimensional numpy.nd.array
    :param label_path: SemanticKITTI .label file path
    :param remap_lut: look-up table based on the SemanticKITTI learning_map
    :return: point-wise ground truth label array
    '''
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)

def get_labels(label_path, config):
    '''
    Reads .label file obtains semantic point-wise labels
    :param label_path: SemanticKITTI .label file path
    :param config: model configuration dictionary
    :return: point-wise ground truth label array
    '''
    remap_dict_val = config["learning_map"]
    max_key = max(remap_dict_val.keys())
    remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
    labels = load_label_kitti(label_path, remap_lut=remap_lut_val)
    return labels

def read_calib_file(filepath):
    '''
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    :param filepath: calibration (.txt) file path
    :return: dictionary containing calibration parameters
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def get_split_files(dataset_path, cfg, count=-1, shuffle=False):
    '''
    Obtains list of files for each dataset split category from the SemanticKITTI dataset base directory
    :param dataset_path: SemanticKITTI dataset base directory
    :param cfg: model configuration dictionary
    :param count: stop index for file processing
    :param shuffle: boolean flag to shuffle file list before returning
    :return:
    '''
    train_seqs = cfg["tr_seq"]
    val_seqs = cfg["va_seq"]
    test_seqs = cfg["te_seq"]

    train_file_list = []
    test_file_list = []
    val_file_list = []
    seq_list = np.sort(listdir(dataset_path))

    for seq_id in seq_list:
        seq_path = join(dataset_path, seq_id)
        pc_path = join(seq_path, 'velodyne')
        if seq_id in train_seqs:
            train_file_list.append([join(pc_path, f) for f in np.sort(listdir(pc_path))[:count]])
        elif seq_id in val_seqs:
            val_file_list.append([join(pc_path, f) for f in np.sort(listdir(pc_path))[:count]])
        elif seq_id in test_seqs:
            test_file_list.append([join(pc_path, f) for f in np.sort(listdir(pc_path)[:count])])

    train_file_list = np.concatenate(train_file_list, axis=0)
    val_file_list = np.concatenate(val_file_list, axis=0)
    test_file_list = np.concatenate(test_file_list, axis=0)

    np.random.seed(234)

    if shuffle:
        np.random.shuffle(train_file_list)
        np.random.shuffle(val_file_list)
        np.random.shuffle(test_file_list)

    return train_file_list, val_file_list, test_file_list


def get_cfg_params(base_dir, dataset_cfg='config/semantic-kitti.yaml', train_cfg='config/tr_config.yml'):
    '''
    Combines parameters from the SemanticKITTI as well as the user-defined training config files into single dictionary.
    :param base_dir: SemanticKITTI dataset base directory
    :param dataset_cfg: SemanticKITTI configuration file (.yaml)
    :param train_cfg: Training configuration file (.yaml)
    :return: model configuration dictionary
    '''
    semkitti_cfg = safe_load(open(dataset_cfg, 'r'))
    tr_params = safe_load(open(train_cfg, 'r'))['training_params']

    split_params = semkitti_cfg['split']

    seq_list = np.sort(listdir(base_dir))

    tr_dict = {'ep': tr_params['epochs'],
               'num_classes': tr_params['num_classes'],
               'patience': tr_params['es_patience'],
               'batch_size': tr_params['batch_size'],
               'l2_reg': tr_params['l2_reg'],
               'n_node_features': tr_params['n_node_features'],
               'learning_rate': tr_params['learning_rate'],
               'lr_decay': round(tr_params['lr_decay_steps'] * tr_params['epochs']),
               'loss_switch_ep': round(tr_params['lovasz_switch_ratio'] * tr_params['epochs']),
               'tr_seq': list(seq_list[split_params['train']]),
               'va_seq': list(seq_list[split_params['valid']]),
               'te_seq': list(seq_list[split_params['test']]),
               'class_ignore': semkitti_cfg["learning_ignore"],
               'learning_map': semkitti_cfg["learning_map"],
               'learning_map_inv': semkitti_cfg["learning_map_inv"],
               'color_map': np.array(list(semkitti_cfg['color_map'].values()))/255,
               'labels': semkitti_cfg["labels"]}

    return tr_dict