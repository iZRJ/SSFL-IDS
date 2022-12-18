import numpy as np
from torch.utils.data.dataset import TensorDataset
import random
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import torch 

def GetAllFeatureLabel(filenames,label_start_idx):

    all_sample_feature = None
    all_sample_label = None
    for i in range(len(filenames)):
        data = np.loadtxt(filenames[i], delimiter=',', encoding='utf-8-sig')
        cur_feature = data[:, 0:label_start_idx]
        cur_label = np.full((len(cur_feature),), i)
        if all_sample_feature is None:
            all_sample_feature = cur_feature
            all_sample_label = cur_label
        else:
            all_sample_feature = np.concatenate((all_sample_feature, cur_feature), axis = 0 )
            all_sample_label = np.concatenate((all_sample_label, cur_label), axis = 0)
    return all_sample_feature, all_sample_label

def TLFeatureLabel(features, labels, TL):

    if len(labels) % TL == 0:
        a_feature_shape = features.shape[1]
        all_TL_feature = np.reshape(features,(-1,TL,a_feature_shape))
        all_TL_label = []
        for sample_idx in range(0,len(labels),TL):
            all_TL_label.append(labels[sample_idx])
        all_TL_label = np.array(all_TL_label)
        
    else:
        all_TL_feature = []
        all_TL_label = []
        for sample_idx in range(0, len(features), TL):
            cur_sample_feature = []
            cur_sample_label = labels[sample_idx]
            for k in range(TL):
                cur_sample_feature.append(features[sample_idx+k])
            cur_sample_feature = np.array(cur_sample_feature)
            all_TL_feature.append(cur_sample_feature)
            all_TL_label.append(cur_sample_label)
        all_TL_feature = np.array(all_TL_feature)
        all_TL_label = np.array(all_TL_label)
    return all_TL_feature, all_TL_label

def SplitPrivateOpen(features, labels, private_percent, open_percent, class_cat, open_fixed):
    if open_fixed == False:
        shuffle_in_unison(features, labels)
    feature_by_label = []
    for i in range(class_cat):
        feature_by_label.append(deque())
    for i in range(len(labels)):
        cur_label = labels[i]
        feature_by_label[cur_label].append(features[i])
    each_class_cnt = []
    for i in range(class_cat):
        each_class_cnt.append(len(feature_by_label[i]))

    private_set_feature = []
    private_set_label = []
    open_set_feature = []
    open_set_label = []
    for i in range(class_cat):
        private_cnt = int(each_class_cnt[i] * private_percent)
        open_cnt = int(each_class_cnt[i] * open_percent)
        cur_cnt = 0
        while cur_cnt < private_cnt and len(feature_by_label[i]) > 0:
            feature = feature_by_label[i].popleft()
            feature = np.array(feature)
            private_set_feature.append(feature)
            private_set_label.append(i)
            cur_cnt += 1
        cur_cnt = 0
        while cur_cnt < open_cnt and len(feature_by_label[i]) > 0:
            feature = feature_by_label[i].popleft()
            feature = np.array(feature)
            # print(feature.dtype)
            open_set_feature.append(feature)
            open_set_label.append(i)
            cur_cnt += 1
    return private_set_feature, private_set_label, open_set_feature, open_set_label

def FlatTLAugData(feature,each_line_shape):
    target_reshape = (-1,) + (each_line_shape,)
    return np.reshape(feature, target_reshape)

def MinMaxOpenPrivate(private_feature,open_feature):
    scaler = MinMaxScaler()
    scaler.fit(open_feature)
    open_feature = scaler.transform(open_feature)
    private_feature = scaler.transform(private_feature)
    return private_feature, open_feature, scaler

def TLAggData(feature,each_line_shape,TL):
    feature = np.reshape(feature,(-1,TL,each_line_shape))
    return feature 

def SplitPrivate(feature,label,client_cnt,class_cat,iid,average):
    feature = np.array(feature)
    label = np.array(label)
    if (iid == True):
        shuffle_in_unison(feature, label) 
    else:
         feature, label = shuffle_non_iid( feature, label,client_cnt, class_cat, average)
    private_length = len(label)

    start_idx = 0
    split_cnt = private_length // client_cnt
    private_datasets = []
    end_idx = start_idx + split_cnt
    for i in range(client_cnt):
        sub_feature = feature[start_idx:end_idx]
        sub_label = label[start_idx:end_idx]
        sub_feature = np.array(sub_feature)
        sub_label = np.array(sub_label)

        train_dataset = GetDataset(sub_feature, sub_label)
        private_datasets.append(train_dataset)

        start_idx = end_idx
        end_idx += split_cnt
    return private_datasets

def DilSplitPrivate(feature, label, client_cnt, class_cat, alpha, seed):
    alpha = [alpha]*(client_cnt*class_cat)
    probality = np.random.dirichlet(alpha,1).transpose()
    sum_pro = 0
    for i in range(len(probality)):
        sum_pro += probality[i]
    probality = np.reshape(probality,(client_cnt,class_cat))
    total_cnt = len(feature)
    
    que_features, que_labels = Queued(feature, label, class_cat)
    each_label_total_cnt = []
    for i in range(len(que_features)):
        each_label_total_cnt.append(len(que_features[i]))
    private_datasets = []
    for client_id in range(client_cnt):
        cur_client_feature = []
        cur_client_label = []
        cur_client_probality = probality[client_id]
        for label_id in range(class_cat):
            cur_label_cnt = int( total_cnt * cur_client_probality[label_id])
            if cur_label_cnt == 0 and label_id + 1 > len(cur_client_probality):
                cur_client_probality[label_id + 1] += cur_client_probality[label_id]
            added_cnt = 0
            while added_cnt < cur_label_cnt and len(que_features[label_id]) > 0:
                cur_client_feature.append(que_features[label_id].popleft())
                cur_client_label.append(que_labels[label_id].popleft())
                added_cnt += 1
        cur_client_feature = np.array(cur_client_feature)
        cur_client_label = np.array(cur_client_label)
        train_dataset = GetDataset(cur_client_feature, cur_client_label)
        private_datasets.append(train_dataset)
    return  private_datasets

def Queued(features, labels, class_cat):
    sorted_feature = []
    sorted_label = []
    sorted_dict = {}
    for i in range(len(features)):
        cur_label = labels[i]
        if cur_label in sorted_dict:
            sorted_dict[cur_label].append(features[i])
        else:
            sorted_dict[cur_label] = []
            sorted_dict[cur_label].append(features[i])
    queued_feature = []
    queued_label = []
    for i in range(class_cat):
        if i in sorted_dict:
            cur_feature = sorted_dict[i]
            cur_label = [i] * (len(cur_feature))
            cur_feature = deque(cur_feature)
            cur_label = deque(cur_label)
            queued_feature.append(cur_feature)
            queued_label.append(cur_label)
    return queued_feature, queued_label
            

def BinarySplitPrivate(feature,label,client_cnt,class_cat = 2,iid = False):
    features = [None] * class_cat
    labels = [None] * class_cat
    each_label_nums = [0] * class_cat
    each_client_samples_cnt = len(label) / client_cnt
    for i in range(len(label)):
        cur_label = label[i]
        cur_feature = feature[i]
        if labels[cur_label] is not None:
            labels[cur_label].append(cur_label)
            features[cur_label].append(cur_feature)
        else:
            labels[cur_label] = [cur_label]
            features[cur_label] = [cur_feature]
        each_label_nums[cur_label] += 1
    start_idx = [0] * class_cat
    all_datasets = []
    for i in range(client_cnt):
        cur_label = random.randint(0,class_cat-1)
        feature_more,label_more = get_feature_label_in_binary(cur_label,each_client_samples_cnt,0.9,features,labels,start_idx)
        feature_less,label_less = get_feature_label_in_binary((cur_label+1)%2,each_client_samples_cnt,0.1,features,labels,start_idx)
        if feature_more is not None and feature_less is not None:
            cur_client_feature = np.concatenate((feature_more,feature_less),axis = 0)
            cur_client_label = np.concatenate((label_more,label_less),axis = 0)
        elif feature_more is None:
            cur_client_feature = feature_less
            cur_client_label = label_less
        else:
            cur_client_feature = feature_more
            cur_client_label = label_more
        train_dataset = GetDataset(cur_client_feature, cur_client_label)
        all_datasets.append(train_dataset)
    return all_datasets

def get_feature_label_in_binary(cur_label,each_client_samples_cnt,ratio,features,labels,start_idx):
    cur_label_num = int(each_client_samples_cnt * ratio)
    if start_idx[cur_label] + cur_label_num < len(features[cur_label]):
        cur_client_feature = features[cur_label][start_idx[cur_label]:start_idx[cur_label]+cur_label_num]
        cur_client_label = labels[cur_label][start_idx[cur_label]:start_idx[cur_label]+cur_label_num]
    elif start_idx[cur_label] < len(features[cur_label]):
        cur_client_feature = features[cur_label][start_idx[cur_label]:]
        cur_client_label = labels[cur_label][start_idx[cur_label]:]
    else:
        cur_client_feature = None
        cur_client_label = None

    start_idx[cur_label] += cur_label_num
    return cur_client_feature, cur_client_label

def MakeDataset(features, labels, TL, need_min_max = True, need_TL = True, need_Transpose = True):
    if need_min_max :
        features = minMaxTrans(features)
    all_TL_feature = []
    all_TL_label = []

    for sample_idx in range(0, len(features), TL):
        cur_sample_feature = []
        cur_sample_label = labels[sample_idx]

        if need_TL == True:
            for k in range(TL):
                cur_sample_feature.append(features[sample_idx+k])
        else:
            cur_sample_feature = features[sample_idx]
        cur_sample_feature = np.array(cur_sample_feature)

        if need_Transpose:
            cur_sample_feature = np.transpose(cur_sample_feature, (1,0)) # 旋转以适应网络结果
        all_TL_feature.append(cur_sample_feature)
        all_TL_label.append(cur_sample_label)
    all_TL_feature = np.array(all_TL_feature)
    all_TL_label = np.array(all_TL_label)
    shuffle_in_unison(all_TL_feature, all_TL_label)
    all_TL_feature, all_TL_label = torch.FloatTensor(all_TL_feature), torch.LongTensor(all_TL_label)
    dataset = torch.utils.data.TensorDataset(all_TL_feature, all_TL_label)
    return dataset


def SplitByLabel(dataset, private_percent, open_percent, class_cat):
    cat_feature = []
    for i in range(class_cat):
        cat_feature.append(deque())
    for fea, label in dataset:
        cat_feature[label].append(fea)
    cat_cnt = []
    for i in range(class_cat):
        cat_cnt.append(len(cat_feature[i]))
    private_set_feature = []
    private_set_label = []
    open_set_feature = []
    open_set_label = []
    for i in range(class_cat):
        private_cnt = int(cat_cnt[i] * private_percent)
        open_cnt = int(cat_cnt[i] * open_percent)
        cur_cnt = 0
        while cur_cnt < private_cnt and len(cat_feature[i]) > 0:
            feature = cat_feature[i].popleft()
            feature = np.array(feature)
            private_set_feature.append(feature)
            private_set_label.append(i)
            cur_cnt += 1

        cur_cnt = 0
        while cur_cnt < open_cnt and len(cat_feature[i]) > 0:
            feature = cat_feature[i].popleft()
            feature = np.array(feature)
            open_set_feature.append(feature)
            open_set_label.append(i)
            cur_cnt += 1
    return private_set_feature, private_set_label, open_set_feature, open_set_label

def SplitAsMultiDatasets(feature, label, iid, client_cnt, class_cat, average = False):
    print("Split As Datasets......")
    feature = np.array(feature)
    label = np.array(label)
    if (iid == True):
        shuffle_in_unison(feature, label) 
    else:
         feature, label = shuffle_non_iid( feature, label,client_cnt, class_cat, average)
    private_length = len(label)

    start_idx = 0
    split_cnt = private_length // client_cnt

    end_idx = start_idx + split_cnt
    private_datasets = []
    for i in range(client_cnt):
        sub_feature = feature[start_idx:end_idx]
        sub_label = label[start_idx:end_idx]
        sub_feature = np.array(sub_feature)
        sub_label = np.array(sub_label)

        train_dataset = GetDataset(sub_feature, sub_label)
        private_datasets.append(train_dataset)

        start_idx = end_idx
        end_idx += split_cnt

    return private_datasets

def minMaxTrans(data):
    min_max = MinMaxScaler()
    data = min_max.fit_transform(data)
    return data

def shuffle_in_unison(a, b):

    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def ShuffleDataset(dataset):

    feature, label = dataset[:]
    feature, label = np.array(feature), np.array(label)
    shuffle_in_unison(feature,label)
    feature,label = torch.tensor(feature),torch.tensor(label)
    dataset = TensorDataset(feature,label)
    return dataset

def shuffle_non_iid(feature, label, client_cnt, class_cat, average = False):

    feature, label = sorted_by_label(feature, label, class_cat)
    sample_shape = feature[0].shape
    taregt_reshape = (-1,) + sample_shape

    if average == False:
        parts = client_cnt 
    else:
        parts = client_cnt * 2

    num_of_each_part = len(label) // parts
    feature_parts_list = []
    label_parts_list = []
    start_idx = 0
    end_idx = num_of_each_part
    for i in range(parts):
        part_feature = feature[start_idx:end_idx]
        part_label = label[start_idx:end_idx]
        feature_parts_list.append(part_feature)
        label_parts_list.append(part_label)
        start_idx = end_idx
        end_idx += num_of_each_part
    shuffle_in_unison(feature_parts_list, label_parts_list)
    part_feature = np.array(feature_parts_list)
    part_label = np.array(label_parts_list)
    part_feature = np.reshape(part_feature, taregt_reshape)
    part_label = part_label.reshape(-1)
    return part_feature, part_label

def sorted_by_label(feature, label, class_cat):
    sorted_feature = []
    sorted_label = []
    sorted_dict = {}
    for i in range(len(feature)):
        cur_label = label[i]
        if cur_label in sorted_dict:
            sorted_dict[cur_label].append(feature[i])
        else:
            sorted_dict[cur_label] = []
            sorted_dict[cur_label].append(feature[i])
    for i in range(class_cat):
        if i in sorted_dict:
            sorted_feature += sorted_dict[i]
            labels = [i] * len(sorted_dict[i])
            sorted_label += labels
    sorted_feature = np.array(sorted_feature)
    sorted_label = np.array(sorted_label)
    return sorted_feature, sorted_label

   
def GetLoader(data_feature, data_label, batch_size ):
    dataset = GetDataset(data_feature,data_label)
    data_loader = GetLoaderFromDataset(dataset,batch_size)
    return data_loader 

def GetLoaderFromDataset(dataset, batch_size,shuffle = True):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False 
    )
    return data_loader

def GetDataset(feature, label):
    feature, label = torch.FloatTensor(feature), torch.LongTensor(label)
    dataset = torch.utils.data.TensorDataset(feature, label)
    return dataset

def TLAggragate(features, TL):
    all_TL_feature = []
    for sample_idx in range(0, len(features), TL):
        cur_sample_feature = []
        for k in range(TL):
            cur_sample_feature.append(features[sample_idx+k])
        cur_sample_feature = np.array(cur_sample_feature)
        all_TL_feature.append(cur_sample_feature)
    all_TL_feature = np.array(all_TL_feature)
    random.shuffle(all_TL_feature)
    return all_TL_feature

def GetFeatureFromOpenDataset(open_dataset, start_idx, end_idx):
    feature, label = open_dataset[start_idx:end_idx]
    return feature, label

def SplitDataFrame(data_frame, TL,percent1, percent2):
    data_array = np.array(data_frame)
    data_array = TLAggragate(data_array, TL)
    aug_sample_cnt = len(data_array)
    part1_cnt = int(percent1 * aug_sample_cnt)
    part2_cnt = int(percent2 * aug_sample_cnt)
    part1_array = data_array[0:part1_cnt]
    part2_array = data_array[part1_cnt:part1_cnt + part2_cnt]
    return part1_array, part2_array

if __name__ == "__main__":
    pass