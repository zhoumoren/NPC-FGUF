import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.dropout_rate = dropout

        # 第一层多分支GCN
        self.gc11 = GraphConvolution(nfeat, nhid)
        self.gc12 = GraphConvolution(nfeat, nhid)
        self.gc13 = GraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)

        # 第二层多分支GCN
        self.gc21 = GraphConvolution(nhid, nhid)
        self.gc22 = GraphConvolution(nhid, nhid)
        self.gc23 = GraphConvolution(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        # # 第三层多分支GCN
        # self.gc31 = GraphConvolution(nhid, nhid)
        # self.gc32 = GraphConvolution(nhid, nhid)
        # self.gc33 = GraphConvolution(nhid, nhid)
        # self.bn3 = nn.BatchNorm1d(nhid)

        # 特征融合层
        self.fusion = nn.Linear(nhid*3, nhid)
        self.fusion1 = nn.Linear(nhid * 3, nhid)

        # 增强MLP结构
        self.mlp1 = nn.Linear(nhid, nclass)

    def forward(self, x, adj, adj1, adj2):
        # 第一层处理
        x1 = F.relu(self.gc11(x, adj))
        x2 = F.relu(self.gc12(x, adj1))
        x3 = F.relu(self.gc13(x, adj2))
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fusion(x)
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        # 第二层处理
        x1 = F.relu(self.gc21(x, adj))
        x2 = F.relu(self.gc22(x, adj1))
        x3 = F.relu(self.gc23(x, adj2))
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fusion1(x)
        x = self.bn2(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        # # 第三层处理
        # x1 = F.relu(self.gc31(x, adj))
        # x2 = F.relu(self.gc32(x, adj1))
        # x3 = F.relu(self.gc33(x, adj2))
        # x = torch.cat([x1, x2, x3], dim=1)
        # x = self.fusion(x)
        # x = self.bn3(x)
        # x = F.dropout(x, self.dropout_rate, training=self.training)

        # MLP分类
        x = self.mlp1(x)
        return F.log_softmax(x, dim=1)


# class MLP(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.fc2 = nn.Linear(hidden_features, hidden_features)
#         self.fc3 = nn.Linear(hidden_features, out_features)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
# class GCN_MLP(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features, n_classes,dropout):
#         super(GCN_MLP, self).__init__()
#
#         # self.gcn = GraphConvolution(in_features, hidden_features)
#         self.gcn=GCN(in_features,2*in_features,hidden_features,dropout)
#         self.mlp1 = MLP(hidden_features, 64, 32)
#         self.mlp2 = MLP(32, 32, 16)
#         self.mlp3 = MLP(16, 16, n_classes)
#
#     def forward(self, x, adj):
#         x_gcn = self.gcn(x, adj)
#         x_mlp1 = self.mlp1(x_gcn)
#         x_mlp2 = self.mlp2(x_mlp1)
#         x_output = self.mlp3(x_mlp2)
#         return x_output