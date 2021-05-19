from os import listdir
from os.path import join
import numpy as np
import struct
from yaml import safe_load
from sklearn.preprocessing import minmax_scale


def read_bin_velodyne(pc_path, include_intensity=False):
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
            if include_intensity:
                pc_list.append([point[0], point[1], point[2], point[3]])
            else:
                pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float16)


def load_label_kitti(label_path, remap_lut):
    '''
    Reads .label file and maps it to a sparse one-dimensional numpy.nd.array
    :param label_path: SemanticKITTI .label file path
    :param remap_lut: look-up table based on the SemanticKITTI learning_map
    :return: point-wise ground truth label array
    '''
    label = np.fromfile(label_path, dtype=int)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int8)


def get_labels(label_path, config):
    '''
    Reads .label file obtains semantic point-wise labels
    :param label_path: SemanticKITTI .label file path
    :param config: model configuration dictionary
    :return: point-wise ground truth label array
    '''
    remap_dict_val = config["learning_map"]
    max_key = max(remap_dict_val.keys())
    remap_lut_val = np.zeros((max_key + 100), dtype=np.int8)
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
                data[key] = np.array([float(x) for x in value.split()], dtype=np.float16)
            except ValueError:
                pass
    return data


def get_split_files(cfg, count=-1, shuffle=False):
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
    seq_list = np.sort(listdir(cfg['base_dir']))

    for seq_id in seq_list:
        pc_path = join(cfg['base_dir'], seq_id)
        if cfg['name'] == 'semantickitti':
            pc_path = join(pc_path, 'velodyne')
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


def get_cfg_params(cfg_file='config/tr_config.yml'):
    '''
    Combines parameters from the SemanticKITTI as well as the user-defined training config files into single dictionary.
    :param base_dir: dataset base directory
    :param dataset_cfg: dataset configuration file (.yaml)
    :param train_cfg: Training configuration file (.yaml)
    :return: model configuration dictionary
    '''
    cfg = safe_load(open(cfg_file, 'r'))
    tr_params = cfg['training_params']
    base_dir = cfg['dataset']['base_dir']
    dataset_cfg = safe_load(open(cfg['dataset']['config'], 'r'))
    split_params = dataset_cfg['split']

    seq_list = np.sort(listdir(base_dir))

    model_dict = {'epochs': tr_params['epochs'],
                  'num_classes': tr_params['num_classes'],
                  'batch_size': tr_params['batch_size'],
                  'l2_reg': tr_params['l2_reg'],
                  'n_node_features': tr_params['n_node_features'],
                  'learning_rate': tr_params['learning_rate'],
                  'loss_switch_ep': round(tr_params['lovasz_switch_ratio'] * tr_params['epochs']),
                  'name': dataset_cfg['name'],
                  'base_dir': base_dir,
                  'tr_seq': list(seq_list[split_params['train']]),
                  'va_seq': list(seq_list[split_params['valid']]),
                  'te_seq': list(seq_list[split_params['test']]),
                  'class_ignore': dataset_cfg["learning_ignore"],
                  'learning_map': dataset_cfg["learning_map"],
                  'learning_map_inv': dataset_cfg["learning_map_inv"],
                  'learning_label_map': dataset_cfg["learning_label_map"],
                  'color_map': np.array(list(dataset_cfg['color_map'].values())) / 255.,
                  'feature_spec': tr_params['feature_spec'],
                  'labels': dataset_cfg["labels"]}

    return model_dict


def save_summary(model):
    from contextlib import redirect_stdout
    with open('models/summaries/' + model.name + '.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()


if __name__ == '__main__':

    model_cfg = get_cfg_params('../config/tr_config.yml')
