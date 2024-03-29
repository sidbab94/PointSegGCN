import numpy as np

class iouEval:
    """
    Computes Intersection-Over-Union (or Jaccard Index) of predictions.
    IoU is the area of overlap between the predicted segmentation and the ground truth
    divided by the area of union between the predicted segmentation and the ground truth.

    Code reference: https://github.com/PRBonn/semantic-kitti-api
    """

    def __init__(self, n_classes, class_ignore=None):
        # classes
        self.n_classes = n_classes

        if class_ignore is not None:
            # what to ignore for eval
            self.class_ignore = class_ignore
            self.get_ignore()

        # What to include and ignore from the means
        self.ignore = np.array(self.ignore_list, dtype=np.int64)
        self.include = np.array(
            [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
        print("[IOU EVAL] IGNORE: ", self.ignore)
        print("[IOU EVAL] INCLUDE: ", self.include)

        # reset the class counters
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = np.zeros((self.n_classes,
                                     self.n_classes),
                                    dtype=np.int64)

    def addBatch(self, x, y):  # x=preds, y=targets
        # sizes should be matching
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # check
        assert (x_row.shape == x_row.shape)

        # create indexes
        idxs = tuple(np.stack((x_row, y_row), axis=0))

        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.conf_matrix, idxs, 1)

    def getStats(self):
        # remove fp from confusion on the ignore classes cols
        conf = self.conf_matrix.copy()
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"

    def get_confusion(self):
        return self.conf_matrix.copy()

    def get_ignore(self):
        self.ignore_list = []
        for cl, ign in self.class_ignore.items():
            if ign:
                x_cl = int(cl)
                self.ignore_list.append(x_cl)
                print("     Ignoring cross-entropy class ", x_cl, " in IoU evaluation")


if __name__ == "__main__":
    # mock problem
    nclasses = 2
    ignore = []
    # test with 2 squares and a known IOU
    lbl = np.zeros((7, 7), dtype=np.int64)
    argmax = np.zeros((7, 7), dtype=np.int64)

    # put squares
    lbl[2:4, 2:4] = 1
    argmax[3:5, 3:5] = 1

    # make evaluator
    eval = iouEval(nclasses, ignore)

    # run
    eval.addBatch(argmax, lbl)
    m_iou, iou = eval.getIoU()
    print("IoU: ", m_iou)
    print("IoU class: ", iou)
    m_acc = eval.getacc()
    print("Acc: ", m_acc)
