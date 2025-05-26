import numpy as np
import scipy.sparse as sp
import torch
import RAND_idx
import pickle
import pandas as pd

# def encode_onehot(labels):
    # classes = set(labels)
    # classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
    #                 enumerate(classes)}
    # labels_onehot = np.array(list(map(classes_dict.get, labels)),
    #                          dtype=np.int32)
    # print(labels_onehot)
    # return labels_onehot



def load_data(dataset="AAAdata"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    data = pd.read_csv(r"..\2025\POI_join.CSV",encoding='gbk')
    # with open('../AAAdata/features_all.pkl', 'rb') as file:
    #     features = pickle.load(file)
    # 使用pd.get_dummies()函数对数组进行one-hot编码
    # features = pd.get_dummies(AAAdata["中类"])


    # labels = encode_onehot(AAAdata['Level2_1'])  #标签 one-hot label   67870*12
    # labels = encode_onehot(idx_features_labels[:, -1])

    # label=[101,201,202,301,402,403,501,502,503,504,505,0]
    # label_A={element: index for index, element in enumerate(label)}
    # labels=[label_A[ele] for i ,ele in enumerate(AAAdata['Level2'])]
    # labels = [label_A[ele] for i, ele in enumerate(AAAdata['Level1'])]
    # labels=AAAdata['Level2']

    # leibie_dict ={v : i for i ,v in  enumerate(set(data["中类"])) }
    # print(leibie_dict)
    # with open('./data_A/leibie_ZA.pkl', 'wb') as file:
    #     pickle.dump(leibie_dict, file)
    # features = []
    # labels = []
    # for i in range(len(data["中类"])):
    #     feature = np.zeros(len(leibie_dict))
    #     feature[leibie_dict[data['中类'][i]]] = 1
    #     features.append(np.array(feature))
    #     labels.append(leibie_dict[data['中类'][i]])
    with open(r'..\2025\features.pkl', 'rb') as file:
        features = pickle.load(file)
    with open(r'..\2025\label_list.pkl', 'rb') as file:
        labels = pickle.load(file)

    with open(r'..\2025\adj_.pkl', 'rb') as file:
        adj = pickle.load(file)

    with open(r'..\2025\adj_weaken_0.01.pkl', 'rb') as file:
        adj1 = pickle.load(file)

    with open(r'..\2025\adj_reciprocal_0.01 .pkl', 'rb') as file:
        adj2 = pickle.load(file)

    # with open(r'D:\BaiduSyncdisk\城市功能分析框架\论文代码\AAAdata\Complete data\R0001.pkl', 'rb') as file:
    #     R = pickle.load(file)

    # 确保数据维度一致
    n_samples = features.shape[0]  # 使用实际有效样本数
    assert len(labels) == n_samples, "特征与标签样本数不一致"

    # adj=normalize(s)
    # s=normalize(s)
    # adj = normalize(adj.dot(s))+ sp.eye(adj.shape[0])
    # adj=normalize(adj)
    # R=R+ sp.eye(R.shape[0])
    adj = adj + sp.eye(adj.shape[0])
    adj1 = adj1 + sp.eye(adj1.shape[0])
    adj2 = adj2 + sp.eye(adj2.shape[0])
    # adj = R
    # adj = normalize(adj)

    idx_train,idx_val,idx_test = RAND_idx.get_random_idx()
    idx_test = range(len(data["中类"]))
    # print(features.shape)


    # features = torch.FloatTensor(np.array(features.todense()))

    #labels = torch.LongTensor(np.where(labels)[1])
    labels=torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.toarray()))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    adj1 = sparse_mx_to_torch_sparse_tensor(adj1)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj2)

    return adj, adj1, adj2, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))            #矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  #求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)             #构造对角矩阵
    mx = r_mat_inv.dot(mx)                  #构造D-1^A
    return mx



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# matrix = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(normalize(matrix))