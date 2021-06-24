import os
from pathlib import PurePath, Path
import numpy as np
from yaml import safe_load
from PIL import Image


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

    # seq_list = np.sort(os.listdir(base_dir))


    model_cfg = {**tr_params,
                 'name': dataset_cfg['name'],
                 'base_dir': base_dir,
                 'tr_seq': split_params['train'],
                 'va_seq': split_params['valid'],
                 'te_seq': split_params['test'],
                 'class_ignore': dataset_cfg["learning_ignore"],
                 'learning_map': dataset_cfg["learning_map"],
                 'learning_map_inv': dataset_cfg["learning_map_inv"],
                 'learning_label_map': dataset_cfg["learning_label_map"],
                 'color_map': np.array(list(dataset_cfg['color_map'].values())) / 255.,
                 'labels': dataset_cfg["labels"]}


    return model_cfg


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
    seq_list = np.sort(os.listdir(cfg['base_dir']))

    for seq_id in seq_list:
        pc_path = os.path.join(cfg['base_dir'], seq_id)
        if cfg['name'] == 'semantickitti':
            pc_path = os.path.join(pc_path, 'velodyne')
        if seq_id in train_seqs:
            train_file_list.append([os.path.join(pc_path, f) for f in np.sort(os.listdir(pc_path))[:count]])
        elif seq_id in val_seqs:
            val_file_list.append([os.path.join(pc_path, f) for f in np.sort(os.listdir(pc_path))[:count]])
        elif seq_id in test_seqs:
            test_file_list.append([os.path.join(pc_path, f) for f in np.sort(os.listdir(pc_path)[:count])])

    train_file_list = np.concatenate(train_file_list, axis=0)
    val_file_list = np.concatenate(val_file_list, axis=0)
    test_file_list = np.concatenate(test_file_list, axis=0)

    np.random.seed(234)

    if shuffle:
        np.random.shuffle(train_file_list)
        np.random.shuffle(val_file_list)
        np.random.shuffle(test_file_list)

    return train_file_list, val_file_list, test_file_list


def read_label_kitti(label_path, config):
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

    remap_dict_val = config["learning_map"]
    max_key = max(remap_dict_val.keys())
    remap_lut_val = np.zeros((max_key + 100), dtype=np.int8)
    remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

    sem_label = remap_lut_val[sem_label]

    return sem_label.astype(np.int8)


def read_calib_txt(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
            except ValueError:
                pass
    return data


def read_scan_attr(file_path, cfg):
    '''
    Encapsulates individual reader functions and accesses all scan attributes
    (point cloud, labels, calibration data, RGB image) from provided point cloud file path.
    :param file_path: point cloud file path
    :param cfg: model config dictionary
    :return: tuple of scan attributes as NumPy arrays
    '''

    path_parts = PurePath(file_path).parts

    if cfg['fwd_pass_check']:
        scan_dir = Path(file_path).parent.absolute()
        label_path = os.path.join(scan_dir, 'y.label')
        calib_path = os.path.join(scan_dir, 'calib.txt')
        img_path = os.path.join(scan_dir, 'img.png')
    else:
        if cfg['name'] == 'semantickitti':
            scan_no = (path_parts[-1]).split('.')[0]
            seq_path = list(path_parts)[:-2]
            seq_path = os.path.join(*seq_path)
            label_path = os.path.join(seq_path, 'labels', scan_no + '.label')
            calib_path = os.path.join(seq_path, 'calib.txt')
            img_path = os.path.join(seq_path, 'image_2', (scan_no + '.png'))

    print(file_path)
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    labels = read_label_kitti(label_path, cfg)
    calib = read_calib_txt(calib_path)
    img = np.asarray(Image.open(img_path))

    return pc, labels, calib, img