import os
import sys
import urllib.request
import pandas as pd
import json

import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
# from sklearn.model_selection import train_test_split

import stellargraph as sg
from stellargraph.layer import GCN_LSTM

import warnings

warnings.filterwarnings('ignore')


def load_data():
    dataset = pd.read_csv('dataset2.csv', index_col=0)
    # dataset = pd.read_csv('dataset1.csv', index_col=0)
    second_indus_list = list(dataset.index)

    second_indus_dict = json.load(open('indus.json', 'r'))
    # 删除数据不全的行业
    del second_indus_dict['CI005029.WI']
    del second_indus_dict['CI005030.WI']

    # 根据二级行业所属的一级行业制作邻接矩阵，所属同一个二级行业的一级行业是相连的
    adj_df = pd.DataFrame(np.zeros((len(second_indus_list), len(second_indus_list))), index=second_indus_list,
                          columns=second_indus_list)
    for first_indus_code, second_indus_code_list in second_indus_dict.items():
        size = len(second_indus_code_list)
        for i in range(size):
            for j in range(i + 1, size):
                adj_df.loc[second_indus_code_list[i], second_indus_code_list[j]] = 1
    adj_symmetric_df = adj_df + adj_df.T + np.eye(len(adj_df))
    adj_second_indus = adj_symmetric_df
    date_list = list(dataset.columns)

    print(str(dataset.shape[0]) + " second industries")
    print(str(dataset.shape[1]) + " days' data")
    print("dataset.shape:", dataset.shape)
    print("adj_second_indus.shape", adj_second_indus.shape)
    return dataset, adj_second_indus, date_list


def sequence_data_preparation(seq_len, pre_len, dataset, date_list, SCALE=True, output='return'):
    '''

    :param seq_len: X的长度
    :param pre_len: 预测pre_len的收益率
    :param dataset: 完整数据集
    :param date_list: 记录日期
    :param SCALE: 是否标准化
    :param output: 输出是收益率还是价格
    :return: 返回X,Y,X的样本所对应的时间，Y所对应的收益率时间
    '''
    X, Y = [], []
    X_time_list = []
    Y_time_list = []
    for i in range(dataset.shape[1] - int(seq_len + pre_len)):
        #     for i in range(dataset.shape[1] - int(seq_len)):
        a = dataset.iloc[:, i: i + seq_len].values
        X_time_list.append(date_list[i] + '-' + date_list[i + seq_len])

        if output == 'return':
            b = (dataset.iloc[:, i + seq_len + pre_len] / dataset.iloc[:, i + seq_len + 1] - 1).values
            Y_time_list.append(date_list[i + seq_len + 1] + '-' + date_list[i + seq_len + pre_len])
        elif output == 'price':
            b = dataset.iloc[:, i + seq_len + 1]

        if SCALE:
            # 时间序列数据如何标准化？
            # 在每个时间窗口上进行标准化，既要对X进行标准化，也要对Y进行标准化
            # axis=1为对每个行业沿着时间进行标准化
            a = scale(a, axis=1)
            b = StandardScaler().fit_transform(b.reshape(-1, 1)).reshape(1, -1)[0]

        X.append(a)
        Y.append(b)

    X = np.array(X)
    Y = np.array(Y)

    print(str(X.shape[0]) + " samples")
    print(str(X.shape[1]) + " second industries")
    print(str(X.shape[2]) + " time step")
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)

    return X, Y, X_time_list, Y_time_list


def train_test_split(X, Y, train_portion=0.8):
    time_len = X.shape[0]
    train_size = int(time_len * train_portion)
    train_X = X[:train_size, :, :]
    train_Y = Y[:train_size, :]
    val_X = X[train_size:, :, :]
    val_Y = Y[train_size:, :]
    return train_X, train_Y, val_X, val_Y


def calculate_index(dataset):
    dataset = dataset.T
    num_date = dataset.shape[0]
    index_list = [np.mean(dataset.iloc[i, :] / dataset.iloc[0, :]) for i in range(num_date)]
    return index_list[-1]


def strategy(X, Y, X_time_list, Y_time_list, dataset, end, model, strategy_top_num=30):
    # 回测策略
    # 在Y_time_list[end].split('-')[0]处买入
    # 在Y_time_list[end].split('-')[1]处卖出
    strategy_top_num = 30
    strategy_X = X[end, :, :]
    strategy_Y = Y[end, :]
    print(f"\nstrategy_X is on {X_time_list[end]}, strategy_Y is on {Y_time_list[end]}")
    strategy_X = np.reshape(strategy_X, (1, dataset.shape[0], seq_len))
    strategy_Y = np.reshape(strategy_Y, (1, dataset.shape[0]))
    strategy_pred_Y = model.predict(strategy_X)
    # 买入日期和卖出日期
    strategy_start_date = Y_time_list[end].split('-')[0]
    strategy_end_date = Y_time_list[end].split('-')[1]
    print(f"Buy date is {strategy_start_date}, sell date is {strategy_end_date}")
    sub_dataset = pd.DataFrame({'second_indus_code': list(dataset.index), 'pred_ratio': strategy_pred_Y[0]})
    sub_dataset = sub_dataset.sort_values(by='pred_ratio', ascending=False)
    # 选前30个行业
    top_indus_list = list(sub_dataset['second_indus_code'].iloc[:strategy_top_num, ])
    # top只取部分行业
    top_indus_index = calculate_index(dataset.loc[top_indus_list, strategy_start_date:strategy_end_date])
    # all取所有行业
    all_indus_index = calculate_index(dataset.loc[:, strategy_start_date:strategy_end_date])
    print(f"Strategy return ratio    is {top_indus_index}")
    print(f"Indus index return ratio is {all_indus_index}")
    return Y_time_list[end], top_indus_index, all_indus_index


# 创建文件夹保存figure
if not os.path.exists('figure'):
    os.makedirs('figure')

# 创建文件夹保存model
if not os.path.exists('model'):
    os.makedirs('model')

seq_len = 50
pre_len = 20
dataset, adj_second_indus, date_list = load_data()
X, Y, X_time_list, Y_time_list = sequence_data_preparation(seq_len=seq_len, pre_len=pre_len,
                                                           dataset=dataset, date_list=date_list)

# 设置一些参数
X_size = len(X)
train_portion = 0.9
best_val_loss = 999999
PATIENCE = 5
EPOCHS = 1000
wait = 0
strategy_top_num = 30

# 带final的是每轮训练的最终结果,_list_list存储训练过程
train_final_loss_list = []
train_final_mse_list = []
train_final_ic_list = []
train_loss_list_list = []
train_mse_list_list = []
train_ic_list_list = []

val_final_loss_list = []
val_final_mse_list = []
val_final_ic_list = []
val_loss_list_list = []
val_mse_list_list = []
val_ic_list_list = []

# 保存每次测试的记录，方便直观查看
pred_test_list = []
true_test_list = []
date_test_list = []

# 滚动的数据集的起止日期和变动步长，step是测试集大小，每次测试结束后要把测试集加入训练集中，再继续训练
start = 0
end = 250
step = 20
tick = 1

# 记录所有结果
res_df = pd.DataFrame(columns=['train_val_date', 'test_date',
                               'train_loss', 'val_loss', 'ave_test_loss',
                               'train_mse', 'val_mse', 'ave_test_mse',
                               'train_ic', 'val_ic', 'ave_test_ic', ])
strategy_res_df = pd.DataFrame(columns=['strategy_date', 'top_indus_ratio', 'all_indus_ratio'])

while end + step < X_size:

    print("\nTick:", tick)

    gcn_lstm = GCN_LSTM(
        seq_len=seq_len, # seq_len – No. of LSTM cells
        adj=adj_second_indus, # adj – unweighted/weighted adjacency matrix of [no.of nodes by no. of nodes dimension
        gc_layer_sizes=[seq_len], #  [seq_len] gc_layer_sizes (list of int) – Output sizes of Graph Convolution layers in the stack.
        gc_activations=["relu"], # ["relu"] gc_activations (list of str or func) – Activations applied to each layer’s output
        lstm_layer_sizes=[100], # lstm_layer_sizes (list of int) – Output sizes of LSTM layers in the stack.
        lstm_activations=["relu"], # lstm_activations (list of str or func) – Activations applied to each layer’s output;
        dropout=0.2, # 0.2
        kernel_initializer="he_normal" # he_normal
    )
    x_input, x_output = gcn_lstm.in_out_tensors()

    model = Model(inputs=x_input, outputs=x_output)
    model.compile(optimizer=Adam(lr=0.0001), loss="mae", metrics=["mse"])
    
    model.summary()

    # 划分训练，验证集，测试集在后面
    sub_X = X[start:end, :, :]
    sub_Y = Y[start:end, :]
    train_X, train_Y, val_X, val_Y = train_test_split(sub_X, sub_Y, train_portion)

    # 保存一次训练的训练过程
    train_loss_list = []
    train_mse_list = []
    train_ic_list = []

    val_loss_list = []
    val_mse_list = []
    val_ic_list = []

    # 做一次训练
    for epoch in range(EPOCHS):
        model.fit(train_X, train_Y, epochs=1, shuffle=False, verbose=0)

        #         print("Epoch:" + str(epoch))

        # 训练集loss和metrics
        pred_train = model.predict(train_X)
        train_loss = mean_absolute_error(pred_train, train_Y)
        trian_mse = mean_squared_error(pred_train, train_Y)
        train_loss_list.append(train_loss)
        train_mse_list.append(trian_mse)
        #         print("train_loss:" + str(train_loss) + " train_mse:" + str(trian_mse),end=" ")

        # 验证集loss和metrics
        pred_val = model.predict(val_X)
        val_loss = mean_absolute_error(pred_val, val_Y)
        val_mse = mean_squared_error(pred_val, val_Y)
        val_loss_list.append(val_loss)
        val_mse_list.append(val_mse)
        #         print("val_loss:" + str(val_loss) +  " val_mse:" + str(val_mse))

        # 训练集ic
        ic_list = []
        for i in range(train_Y.shape[0]):
            ic = spearmanr(train_Y[i], pred_train[i])[0]
            ic_list.append(ic)
        train_ic_mean = np.mean(ic_list)
        train_ic_list.append(train_ic_mean)
        #         print("mean ic on train set:", train_ic_mean, end=" ")

        # 验证集ic
        ic_list = []
        for i in range(val_Y.shape[0]):
            ic = spearmanr(val_Y[i], pred_val[i])[0]
            ic_list.append(ic)
        val_ic_mean = np.mean(ic_list)
        val_ic_list.append(val_ic_mean)
        #         print("mean ic on val set:", val_ic_mean)

        # Early stopping
        if val_loss < best_val_loss:
            # 记录最好的情况
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_model = model
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                best_val_loss = 999999
                wait = 0
                #                 print('Epoch {}: early stopping'.format(epoch))
                #                 print()
                break

    # 保存model
    model.save('model/' + X_time_list[start] + "_to_" + X_time_list[end - 1] + '.h5')

    # 画loss图
    plt.plot(train_loss_list, label="Training loss")
    plt.plot(val_loss_list, label="Val loss")
    plt.legend()
    plt.title("Loss: Train-val samples Date:" + X_time_list[start] + "to" + X_time_list[end - 1])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('figure/loss_' + X_time_list[start] + "_to_" + X_time_list[end - 1] + '.png', dpi=300)
    plt.show()

    # 画ic图
    plt.plot(train_ic_list, label="Training ic")
    plt.plot(val_ic_list, label="Val ic")
    plt.legend()
    plt.title("IC: Train-val samples Date:" + X_time_list[start] + " to " + X_time_list[end - 1])
    plt.xlabel("epoch")
    plt.ylabel("ic")
    plt.savefig('figure/ic_' + X_time_list[start] + "_to_" + X_time_list[end - 1] + '.png', dpi=300)
    plt.show()

    # 加入最后得到的train和val的结果，把训练过程也保存
    train_final_loss_list.append(train_loss)
    train_final_mse_list.append(trian_mse)
    train_final_ic_list.append(train_ic_mean)
    train_loss_list_list.append(train_loss_list)
    train_mse_list_list.append(train_mse_list)
    train_ic_list_list.append(train_ic_list)

    val_final_loss_list.append(val_loss)
    val_final_mse_list.append(val_mse)
    val_final_ic_list.append(val_ic_mean)
    val_loss_list_list.append(val_loss_list)
    val_mse_list_list.append(val_mse_list)
    val_ic_list_list.append(val_ic_list)

    strategy_date, \
    top_indus_ratio, \
    all_indus_ratio = strategy(X, Y, X_time_list, Y_time_list,
                               dataset, end, model, strategy_top_num=30)
    strategy_res_df.loc[tick - 1] = [strategy_date, top_indus_ratio, all_indus_ratio]


    # 测试集，需要取多个测试集，增加generative
    test_loss_list = []
    test_mse_list = []
    test_ic_list = []
    for test_bias in range(step):
        test_X = X[end + test_bias:end + step + test_bias, :, :]
        test_Y = Y[end + test_bias:end + step + test_bias, :]

        # 测试集loss和metrics
        pred_test = model.predict(test_X)
        test_loss = mean_absolute_error(pred_test, test_Y)
        test_mse = mean_squared_error(pred_test, test_Y)
        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)
        #         print("test_loss:" + str(test_loss) +  " test_mse:" + str(test_mse))

        # 保存test的预测和真实记录，以及相对应的时间
        pred_test_list.append(pred_test)
        true_test_list.append(test_Y)
        #         date_test_list.append( date_list[end] + "-" + date_list[end+step-1])
        date_test_list.append(end + test_bias)

        # 测试集集ic
        ic_list = []
        for i in range(test_Y.shape[0]):
            ic = spearmanr(test_Y[i], pred_test[i])[0]
            ic_list.append(ic)
        test_ic = np.mean(ic_list)
        test_ic_list.append(test_ic)
    #         print("mean ic on test set:", test_ic)
    #         print()
    test_loss_mean = np.mean(test_loss_list)
    test_mse_mean = np.mean(test_mse_list)
    test_ic_mean = np.mean(test_ic_list)

    # 记录
    print("Train-val samples Date:", X_time_list[start], " to ", X_time_list[end - 1])
    print("Testset   samples Date:", X_time_list[end], " to ", X_time_list[end + step - 1])
    values = [[train_loss, trian_mse, train_ic_mean], [val_loss, val_mse, val_ic_mean],
              [test_loss_mean, test_mse_mean, test_ic_mean]]
    print(pd.DataFrame(values, index=['Train', 'Val', 'Test'], columns=['loss', 'mse', 'ic']))
    print('\n')

    # 最终的结果写入DataFrame
    res_df.loc[tick - 1, :] = [X_time_list[start] + " " + X_time_list[end - 1],
                               X_time_list[end] + " " + X_time_list[end + step - 1],
                               train_loss, val_loss, test_loss_mean,
                               trian_mse, val_mse, test_mse_mean,
                               train_ic_mean, val_ic_mean, test_ic_mean]

    # 更新训练集，验证集，测试集
    end += step
    tick += 1


res_df.to_csv('record.csv')
res_df.iloc[:, 2:].astype(np.float).describe().to_csv('record_describe.csv')
strategy_res_df.to_csv('strategy_res_df.csv')
