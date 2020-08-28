import pandas as pd
import scipy.sparse as sp
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import pickle
import json
import networkx as nx

import tensorflow.keras.backend as K
from tensorflow.keras import layers, optimizers, losses, metrics, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping

import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from stellargraph import datasets

from sklearn import preprocessing, feature_extraction, model_selection
from scipy import stats
from IPython.display import display, HTML
from create_dataset import load_data


# 定义model.compile()时的metrics
def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)


def train(G_list, nodes_subjects_list, run_num=1, start_month_id=220, end_month_id=264):
    # 提前定义一些列表方便记录数据，大循环的列表存小循环的列表
    graph_history_list_list = []
    model_list_list = []
    train_gen_list_list = []
    time_list_list = []
    model_weight_list_list = []

    # 选择运行run_num次
    run_num = run_num
    # 选择进行训练的月份,end_month_id最多取
    start_month_id = start_month_id
    end_month_id = end_month_id

    # 创建文件夹保存model
    if not os.path.exists('model'):
        os.makedirs('model')

    # 创建文件夹保存history
    if not os.path.exists('history'):
        os.makedirs('history')

    # 创建文件夹保存figure
    if not os.path.exists('figure'):
        os.makedirs('figure')

    # 创建文件夹保存figure
    if not os.path.exists('figure_distribution'):
        os.makedirs('figure_distribution')

    # 创建文件夹保存test结果
    if not os.path.exists('test_result'):
        os.makedirs('test_result')

    # 大循环记录训练了几次，计算多次是为了减少variance
    # 小循环记录训练的月份
    for j in range(run_num):
        num_samples = [40]

        # 提前定义一些列表记录小循环的数据
        graph_history_list = []
        model_list = []
        train_gen_list = []
        time_list = []
        model_weight_list = []
        test_result = []

        # i为0代表220
        for i in range(start_month_id - 220, end_month_id - 220):
            start = time.time()

            # 前一个月训练，后一个月验证
            train_idx = i
            val_idx = i + 1
            test_idx = i + 2

            # 用train_idx的数据生成训练集的generator
            generator = GraphSAGENodeGenerator(G=G_list[train_idx], batch_size=len(nodes_subjects_list[train_idx]),
                                               num_samples=num_samples, seed=100)
            train_gen = generator.flow(list(nodes_subjects_list[train_idx].index),
                                       nodes_subjects_list[train_idx].values,
                                       shuffle=False)

            # 生成GraphSAGE模型
            graphsage_model = GraphSAGE(
                layer_sizes=[1], generator=generator, bias=True, aggregator=sg.layer.MeanAggregator, normalize=None
            )

            # 提取输出输出的tensor，用keras来构建模型
            x_inp, x_out = graphsage_model.in_out_tensors()
            #         prediction = layers.Dense(units=1)(x_out)

            # 用val_idx的数据生成验证集的generator
            generator = GraphSAGENodeGenerator(G=G_list[val_idx], batch_size=len(nodes_subjects_list[val_idx]),
                                               num_samples=num_samples, seed=100)
            val_gen = generator.flow(list(nodes_subjects_list[val_idx].index), nodes_subjects_list[val_idx].values)

            # 用test_idx的数据生成验证集的generator
            generator = GraphSAGENodeGenerator(G=G_list[test_idx], batch_size=len(nodes_subjects_list[test_idx]),
                                               num_samples=num_samples, seed=100)
            test_gen = generator.flow(list(nodes_subjects_list[test_idx].index), nodes_subjects_list[test_idx].values)

            # 通过输入输出的tensor构建model
            model = Model(inputs=x_inp, outputs=x_out)
            monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
                                    patience=10, verbose=2, mode='auto',
                                    restore_best_weights=True)
            model.compile(
                optimizer=optimizers.Adam(lr=0.05),
                loss=losses.mean_squared_error,
                metrics=[pearson_r]
            )

            history = model.fit(
                train_gen, epochs=500, validation_data=val_gen, verbose=0, shuffle=False, callbacks=[monitor]
            )

            test_metrics = model.evaluate(test_gen)
            test_result_dict = {}
            print("\n" + str(train_idx + 220) + "'s Test Set: " + str(test_idx + 220) + "'s Metrics:")
            for name, val in zip(model.metrics_names, test_metrics):
                print("\t{}: {:0.4f}".format(name, val))
                test_result_dict[name] = val
            json.dump(test_result_dict,
                      open('test_result/' + str(train_idx + 220) + "_" + str(test_idx + 220) + '.json', 'w'))

            test_preds = model.predict(test_gen)

            end = time.time()

            # 保存一些结果
            graph_history_list.append(history)  # 保存训练过程
            model_list.append(model)  # 保存model
            train_gen_list.append(train_gen)  # 保存train_gen方便之后算中间层的结果
            time_list.append(end - start)  # 保存运行时间
            model_weight_list.append(model.weights)  # 保存model的参数
            test_result.append(test_metrics[1])

            # # 存模型model
            # model.save('model/' + str(train_idx + 220) + "_" + str(val_idx + 220) + '.h5')
            # # 存训练过程history
            # json.dump(history.history,
            #           open('history/' + str(train_idx + 220) + "_" + str(val_idx + 220) + '.json', 'w'))
            # # 存训练过程图片figure
            # sg.utils.plot_history(history)
            # plt.title(str(train_idx + 220) + '->' + str(val_idx + 220))
            # plt.savefig('figure/' + str(train_idx + 220) + "_" + str(val_idx + 220) + '.png')
            # plt.show()
            # 存test的prediction的distribution
            plt.figure(figsize=(5, 10))
            plt.subplot(211)
            plt.hist(test_preds, bins=500)
            plt.title("Distribution of Prediction of " + str(test_idx + 220))
            plt.subplot(212)
            plt.hist(nodes_subjects_list[test_idx].values, bins=500)
            plt.title("Distribution of Origin of " + str(test_idx + 220))
            plt.xlabel("ic=" + str(test_metrics[1]))
            plt.savefig('figure_distribution/distribution-' + str(train_idx + 220) + "_" + str(test_idx + 220) + '.png',
                        dpi=300)
            plt.show()

            print(str(i + 220) + "'s " + str(j + 1) + " run has finished")
            print()

        # 将小循环的数据保存
        graph_history_list_list.append(graph_history_list)
        model_list_list.append(model_list)
        train_gen_list_list.append(train_gen_list)
        time_list_list.append(time_list)
        model_weight_list_list.append(model_weight_list)

        return graph_history_list_list, model_list_list, train_gen_list_list, time_list_list, model_weight_list_list, test_result


# 计算图产生的因子和每个输入因子的spearnman系数
def cal_ic_between_factor(graph_history_list_list, model_list_list, train_gen_list_list,
                          time_list_list, model_weight_list_list, nodes_features_list,
                          run_num=1, start_month_id=220, end_month_id=225):
    for k in range(run_num):
        graph_history_list = graph_history_list_list[k]
        model_list = model_list_list[k]
        train_gen_list = train_gen_list_list[k]
        time_list = time_list_list[k]
        model_weight_list = model_weight_list_list[k]
        for i in range(start_month_id - 220, end_month_id - 220):
            # 计算reshape层的输出，如果normalization取None的话，那么reshape层输出就和lambda一样
            model = model_list[i]
            graph_layer_model = Model(inputs=model.input,
                                      outputs=model.layers[-2].output)
            graph_output = graph_layer_model.predict(train_gen_list[i])
            print(str(220 + i) + '->' + str(220 + i + 1))

            for col in nodes_features_list[0].columns:
                print(col, stats.spearmanr(graph_output, nodes_features_list[i][col])[0])
            print()


def plot_train_process(graph_history_list_list, model_list_list, train_gen_list_list,
                       time_list_list, model_weight_list_list, nodes_features_list,
                       run_num=1, start_month_id=220, end_month_id=225):
    # 绘制训练时loss和spearnman系数的变化过程
    for k in range(run_num):
        graph_history_list = graph_history_list_list[k]
        model_list = model_list_list[k]
        train_gen_list = train_gen_list_list[k]
        time_list = time_list_list[k]
        model_weight_list = model_weight_list_list[k]
        for i in range(start_month_id - 220, end_month_id - 220):
            # 计算reshape层的输出，如果normalization取None的话，那么reshape层输出就和lambda一样
            sg.utils.plot_history(graph_history_list[i])
            plt.title(str(220 + i) + '->' + str(220 + i + 1))
            plt.savefig('figure/' + str(i + 220) + "_" + str(i + 220 + 1) + '.png')
            plt.show()


def display_model_weights(graph_history_list_list, model_list_list, train_gen_list_list,
                          time_list_list, model_weight_list_list, nodes_features_list,
                          run_num=1, start_month_id=220, end_month_id=225):
    for k in range(run_num):
        graph_history_list = graph_history_list_list[k]
        model_list = model_list_list[k]
        train_gen_list = train_gen_list_list[k]
        time_list = time_list_list[k]
        model_weight_list = model_weight_list_list[k]
        for i in range(start_month_id - 220, end_month_id - 220):
            # 计算reshape层的输出，如果normalization取None的话，那么reshape层输出就和lambda一样
            model = model_list[i]
            print(str(start_month_id) + "->" + str(start_month_id + 1))
            print(model.weights)
            print()


if __name__ == '__main__':
    run_num = 1
    start_month_id = 220
    end_month_id = 230

    # 加载训练数据
    G_list, nodes_subjects_list, nodes_features_list = load_data(start_month_id, end_month_id)

    # 训练，输出训练保存的结果
    graph_history_list_list, model_list_list, train_gen_list_list, time_list_list, model_weight_list_list, test_result = \
        train(G_list, nodes_subjects_list, run_num, start_month_id, end_month_id)

    print(np.mean(test_result))

    # # 这些函数中很多参数是冗余的，但为了方便就不去掉了
    # # 计算ic
    # cal_ic_between_factor(graph_history_list_list, model_list_list, train_gen_list_list,
    #                       time_list_list, model_weight_list_list, nodes_features_list,
    #                       run_num, start_month_id, end_month_id)
    # # 画图
    # plot_train_process(graph_history_list_list, model_list_list, train_gen_list_list,
    #                    time_list_list, model_weight_list_list, nodes_features_list,
    #                    run_num, start_month_id, end_month_id)
    # # 模型参数
    # display_model_weights(graph_history_list_list, model_list_list, train_gen_list_list,
    #                       time_list_list, model_weight_list_list, nodes_features_list,
    #                       run_num, start_month_id, end_month_id)
