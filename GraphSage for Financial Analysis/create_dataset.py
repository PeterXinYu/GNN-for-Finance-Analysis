import pandas as pd
import scipy.sparse as sp
import os
import stellargraph as sg


# 去除因子为缺失值的股票
def create_mat(adj, df_path):
    adjMatDf = adj

    df = pd.read_csv(df_path, index_col=0)
    df = df.dropna(how='any')
    stockCodeAr = df['stock'].values

    commenStockIdLi = list(set(adjMatDf.index).intersection(set(stockCodeAr)))

    adjMatDf = adjMatDf.loc[commenStockIdLi, commenStockIdLi]
    adjMatDf = adjMatDf.sort_index(level=stockCodeAr).sort_index(axis=1, level=stockCodeAr)
    #     adjMatDf[adjMatDf > 0] = 1
    adjMatDf = adjMatDf.dropna(axis=0, how='all').dropna(axis=1, how='all')

    df = df.set_index('stock').loc[adjMatDf.index]

    return adjMatDf, df


def load_data(start_month_id=220, end_month_id=264):


    # # 每个时间点股票的列表的列表，为求这段时间内共同的不含缺失值的股票做准备
    # stock_code_list_list = []
    # for month_id in range(250, 260):
    #     stock_code_list_list.append(list(pd.read_csv('csv_demo_con/' +
    #                                                  str(month_id + 1) + '.csv',index_col = 0).dropna(how='any')['stock'].values))
    # # 求这个时间段里都存在的股票
    # common_stock_code_list = stock_code_list_list[0]
    # for i in range(260-250):
    #     common_stock_code_list = [x for x in common_stock_code_list if x in stock_code_list_list[i]]

    if not os.path.exists('data'):
        os.makedirs('data')

    '''
    G是stellargraph中封装好的对象，存储图相关的信息
    nodes_subjects是节点的标签，这里是norm_return
    nodes_features是节点的特征，这里是因子信息
    提前做好列表，方便存每个月的数据
    '''
    G_list = []
    nodes_subjects_list = []
    nodes_features_list = []

    # range调整需要计算的月份，但是延迟一个月，219实际计算的是220.csv，264实际计算的是265.csv
    # 文件中有220.csv到264.csv，所以range取(219, 264)即可全部读取
    for month_id in range(start_month_id - 1, end_month_id + 1):

        # if如果不存这些文件，就说明是第一次运行，则要计算邻接矩阵adj和因子矩阵factor
        # else如果已经存在，那么直接读取即可
        if not (os.path.exists("data/factor-" + str(month_id + 1) + ".csv") and os.path.exists(
                "data/adj-" + str(month_id + 1) + ".npz")):
            # 行业为0~30，读取为ori_df，为创建行业的邻接矩阵做准备
            ori_df = pd.read_excel('monthly_indus.xlsx', index_col=0, header=None)
            stock_code_list = list(ori_df.index)

            # 字典存stock_code对应的行数
            stock_code_id_dict = {}
            for i, j in zip([i for i in range(3945)], stock_code_list):
                stock_code_id_dict[j] = i

            sub_ori_df = ori_df.iloc[:, month_id]

            # 这三个数组为构建稀疏矩阵做准备
            adj_row = []
            adj_col = []
            adj_data = []

            # 行业编号有30个：0~30
            for indus_code in range(31):
                sub_ori_df_of_indus_code = sub_ori_df[sub_ori_df == indus_code]
                index_of_sub_ori_df_of_indus_code = sub_ori_df_of_indus_code.index
                size = len(sub_ori_df_of_indus_code)
                if size > 1:
                    for i in range(size):
                        for j in range(i, size):
                            # 若为range(i + 1, size)，则邻接矩阵的对角线为0
                            stock_1_code = index_of_sub_ori_df_of_indus_code[i]
                            stock_2_code = index_of_sub_ori_df_of_indus_code[j]
                            stock_1_id = stock_code_id_dict[stock_1_code]
                            stock_2_id = stock_code_id_dict[stock_2_code]
                            adj_row.append(stock_1_id)
                            adj_col.append(stock_2_id)
                            adj_data.append(1)

            # 直接创建稀疏的adj可以大幅加速，但还要转回dataframe，因为要根据stock_code和因子求交集
            adj_sparse = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=(3945, 3945))
            adj_df = pd.DataFrame((adj_sparse).toarray(), index=stock_code_list, columns=stock_code_list)

            # 新得到adj_df, factor_df，里面包含的股票一致
            adj_df, factor_df = create_mat(adj_df, 'csv_demo_con/' + str(month_id + 1) + '.csv')

            # 将adj_df转回adj_sparse，创建edges为创建G对象做准备
            adj_sparse = sp.coo_matrix(adj_df.values)

            # 将邻接矩阵和因子矩阵保存，以后就不用再计算一遍了
            sp.save_npz('data/adj-' + str(month_id + 1) + '.npz', adj_sparse)
            factor_df.to_csv('data/factor-' + str(month_id + 1) + '.csv')

        else:
            adj_sparse = sp.load_npz('data/adj-' + str(month_id + 1) + '.npz')
            factor_df = pd.read_csv('data/factor-' + str(month_id + 1) + '.csv', index_col=0)

        '''
            如
                row = [1, 3, 4, 6, 8]
                col = [3, 5, 7, 8, 10]
            则1节点和3节点有连结，3节点和5节点有连接，以此类推
            边权由weight对应的列表表示
            注意：每个节点是对应不同的股票的
        '''

        row = adj_sparse.row
        col = adj_sparse.col
        edges = pd.DataFrame({
            "source": row,
            "target": col,
            "weight": [1 for i in range(len(row))]
        })

        '''
            nodes是一个列表，第0个特征对应第0个节点，第1个特征对应第1个节点，以此类推
        '''
        nodes = factor_df.reset_index().loc[:, 'return_1m':'return_12m']
        nodes_features_list.append(nodes)

        #     # 大家共同使用第一个月的邻接矩阵
        #     if month_id == 250:
        #         common_edges = edges

        # 创建包含图信息的对象G
        G = sg.StellarGraph(nodes, edges)
        G_list.append(G)

        # 创建每个节点对应的标签，这里是norm_return
        node_subjects = factor_df.reset_index()['norm_return']
        nodes_subjects_list.append(node_subjects)

        print(month_id + 1, "has finished")
    return G_list, nodes_subjects_list, nodes_features_list


if __name__ == '__main__':
    G_list, nodes_subjects_list, nodes_features_list = load_data()
