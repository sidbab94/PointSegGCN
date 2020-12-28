from tensorflow.keras.layers import Dropout, BatchNormalization, Input, Add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from spektral.layers import GCNConv

def conv_relu_bn(parents, filters, id, dropout=False, l2_reg=0.01):
    X_in, A_in = parents
    x = GCNConv(filters, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_' + str(id))([X_in, A_in])
    x = BatchNormalization(name='bn_' + str(id))(x)
    if dropout:
        x = Dropout(0.1, name='do_' + str(id))(x)
    return x

def res_model_1(tr_params):

    l2_reg = tr_params['l2_reg']
    F = tr_params['n_node_features']
    num_classes = tr_params['num_classes']

    X_in = Input(shape=(F, ), name='X_in')
    A_in = Input(shape=(None,), sparse=True)

    X_1 = GCNConv(64, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_1')([X_in, A_in])

    X_2 = conv_relu_bn((X_1, A_in), 64, 2, dropout=True, l2_reg=l2_reg)
    X_3 = conv_relu_bn((X_2, A_in), 64, 3, dropout=True, l2_reg=l2_reg)
    X_4 = conv_relu_bn((X_3, A_in), 64, 4, dropout=True, l2_reg=l2_reg)

    X_5 = Add(name='add_4')([X_4, X_2])

    X_6 = GCNConv(64, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_8')([X_5, A_in])

    X_7 = Add(name='add_6')([X_6, X_1])

    output = GCNConv(num_classes, activation='softmax', name='gcn_6')([X_7, A_in])

    model = Model(inputs=[X_in, A_in], outputs=output, name='GraphSEG_v2')
    return model


def fcn_1(tr_params):

    l2_reg = tr_params['l2_reg']
    F = tr_params['n_node_features']
    num_classes = tr_params['num_classes']

    X_in = Input(shape=(F, ), name='X_in')
    A_in = Input(shape=(None,), sparse=True)

    X_1 = conv_relu_bn((X_in, A_in), 32, 1, dropout=True)
    X_2 = conv_relu_bn((X_1, A_in), 32, 2, dropout=True)
    X_3 = conv_relu_bn((X_2, A_in), 32, 3, dropout=True)
    X_4 = conv_relu_bn((X_3, A_in), 32, 4, dropout=True)

    X_5 = concatenate([X_4, X_3, X_2, X_1], axis=1)

    X_6 = GCNConv(32, activation='relu', kernel_regularizer=l2(l2_reg), name='gcn_5')([X_5, A_in])

    output = GCNConv(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg), name='gcn_6')([X_6, A_in])

    model = Model(inputs=[X_in, A_in], outputs=output, name='GraphFCN_v1')
    return model

