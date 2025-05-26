import pandas as pd
import random

#按0.7.0.15，0.15获取不同类别的poi

def store_indices(data):
    indices_dict = {}

    for index, element in enumerate(data):
        if element==0:
            continue
        if element in indices_dict:
            indices_dict[element].append(index)
        else:
            indices_dict[element] = [index]

    return indices_dict
# def store_indices(AAAdata):
#     indices_dict = {}
#
#     for index, element in enumerate(AAAdata):
#         if element in indices_dict:
#             indices_dict[element].append(index)
#         else:
#             indices_dict[element] = [index]
#     return indices_dict

# 示例列表
data1 = pd.read_csv(r"..\2025\POI_join_join_.CSV",encoding='gbk')
data = data1['中类']
data_t=data1['FID']

# 存储相同元素的下标
indices_dict = store_indices(data)
# print(len(indices_dict))    #97

def split_by_ratio(data, ratios):
    random.shuffle(data)  # 随机打乱列表

    total_length = len(data)
    lengths = [int(total_length * ratio) for ratio in ratios]

    split_lists = []
    start = 0
    for length in lengths:
        end = start + length
        split_lists.append(data[start:end])
        start = end

    return split_lists


#print(matrix3)
def check_duplicates(list1, list2, list3):
    # 将三个列表合并成一个列表
    combined_list = list1 + list2 + list3
    # 使用集合判断是否存在重复元素
    if len(combined_list) == len(set(combined_list)):
        return "没有重复的值"
    else:
        return "存在重复的值"
def transfrom(a):
    b=[]
    for i in a:
        b.append(data_t[i])
    return b



def get_random_idx():
    ratios = [0.7, 0.3, 0]
    idx_train = []
    idx_val = []
    idx_test =[]

    for i in indices_dict:
        a=split_by_ratio(indices_dict[i],ratios)
        # idx_train=random.shuffle(idx_train+a[0])
        # idx_val=random.shuffle(idx_val+a[1])
        # idx_test=random.shuffle(idx_test+a[2])
        idx_train = idx_train + a[0]
        idx_val = idx_val + a[1]
        idx_test = idx_test + a[2]
        # print(idx_train)
        # print(idx_val)
        # print(idx_test)
    print(check_duplicates(idx_train,idx_val,idx_test))
    # idx_train = random.sample(idx_train, len(idx_train))
    # idx_val = random.sample(idx_val, len(idx_val))
    # idx_test = random.sample(idx_test, len(idx_test))
    # print(type(idx_test))
    idx_train=transfrom(idx_train)
    idx_val=transfrom(idx_val)
    idx_test=transfrom(idx_test)
    print(len(idx_train))
    print(len(idx_val))
    print(len(idx_test))

    return idx_train, idx_val, idx_test

#print(sorted(idx_test))

