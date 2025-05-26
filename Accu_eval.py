import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import sklearn.metrics as sm

# io = r"I:\第二个中文\验证.xlsx"
# AAAdata = pd.read_excel(io, sheet_name = 0, header = 0)
# # sheet_name=0 代表读取excle中的第一个sheet，header为定义列名为第0行
# data1 = np.array(AAAdata)

# y_true = data1[:, 0] #实测数据属性列索引号
# y_pred = data1[:, 1] #分类数据属性列索引号



def p_r(y_pred,y_true):
    # print(y_pred)
    # print(y_true)
    # # true positive
    # TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
    # print(TP)
    # # false positive
    # FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    #
    # # true negative
    # TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
    #
    # # false negative
    # FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
    #
    # Precision = TP/(TP + FP)
    # Recall = TP/(TP + FN)
    # Accuracy = (TP + TN)/(TP + FP + TN + FN)
    # Error_rate = (FN + FP)/(TP + FP + TN +FN)
    # F1_score = 2*Precision*Recall/(Precision + Recall)

# po = (TP + TN)/(TP + FP + TN + FN)      #accuracy
# pe = (60*(TP+FP) + 60*(FN+TN))/(120*120) #60为单类样本的个数，120为总样本数量
# Kappa = (po - pe)/(1-pe)
# Confus_matrix = np.array([[FN, FP], [TN, TP]])
    Precision_micro=sm.precision_score(y_true,y_pred,average='micro')
    Precision_none=sm.precision_score(y_true,y_pred,average=None)
    # 计算召回率
    Recall = sm.recall_score(y_true, y_pred,average='micro')
    # 计算总体精度
    Accuracy = sm.accuracy_score(y_true, y_pred)

    # 计算F1分数
    F1_score = sm.f1_score(y_true, y_pred,average='micro')
    Kappa=cohen_kappa_score(y_true, y_pred)

    print("精确率为:micro", Precision_micro)
    print("精确率为:", Precision_none)
    print("召回率为:", Recall)
    print("总体精度为:", Accuracy)
    print("F1分数为:", F1_score)
    print("Kappa系数为:", Kappa)

    Precision_macro = sm.precision_score(y_true, y_pred, average='macro')
    # 计算F1分数
    F1_scoremacro = sm.f1_score(y_true, y_pred, average='macro')
    print("精确率为:macro", Precision_macro)
    print("F1分数为macro:", F1_scoremacro)
# AAAdata=pd.read_csv(r"C:\Users\Lenovo\Desktop\预测MLP.csv",encoding='gbk')
# real=AAAdata['real']
# pre=AAAdata['pre']
# idx=[]
# for i in range(len(real)):
#     if real[i]==11:
#         continue
#     else:
#         idx.append(i)
# y_true=real[idx]
# print(type(y_true))
# y_pred=pre[idx]
# p_r(y_pred,y_true)