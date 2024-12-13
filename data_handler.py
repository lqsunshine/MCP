import h5py
from config import opt
import torch
import numpy as np
import sys
import backdoor
from tqdm import tqdm
import scipy.io as sio

def load_data(path):
    if opt.dataset in ['flickr25k','nus-wide']:
        file = h5py.File(path)
        images = file['IAll'][:]
        labels = file['LAll'][:]
        tags = file['TAll'][:]
        images = images.transpose(3,2,0,1)
        tags = tags.transpose(1,0)
        labels = labels.transpose(1,0)
        file.close()
    else:
        image_path = path + "/image.mat"
        label_path = path + "/label.mat"
        tag_path = path + "/tag.mat"
        images = sio.loadmat(image_path)['Image']  # 19862,224,224,3
        images = images.transpose(0, 3, 1, 2)
        tags = sio.loadmat(tag_path)['Tag']  # 19862,2685
        labels = sio.loadmat(label_path)["Label"]  # 19862,35
    return images, tags, labels

def split_data(images, tags, labels):
    X = {}
    X['query'] = images[0: opt.query_size]
    X['train'] = images[opt.query_size: opt.training_size + opt.query_size]
    X['retrieval'] = images[opt.query_size: opt.query_size + opt.database_size]

    Y = {}
    Y['query'] = tags[0: opt.query_size]
    Y['train'] = tags[opt.query_size: opt.training_size + opt.query_size]
    Y['retrieval'] = tags[opt.query_size: opt.query_size + opt.database_size]

    L = {}
    L['query'] = labels[0: opt.query_size]
    L['train'] = labels[opt.query_size: opt.training_size + opt.query_size]
    L['retrieval'] = labels[opt.query_size: opt.query_size + opt.database_size]

    return X, Y, L

def get_dataset(X, Y, L, is_backdoor = False):
    train_y = torch.from_numpy(Y['train'])
    train_L = None
    train_x = None
    if not is_backdoor:
        train_L = torch.from_numpy(L['train'])
        train_x = torch.from_numpy(X['train'])

    else:
    #poisoning
        temp_train_L = L['train']
        temp_train_x = X['train']
        num_train = temp_train_x.shape[0]

        poisoned_num = int(opt.pr * num_train)
        np.random.seed(10086)
        poisoned_index = np.random.choice(num_train, size=poisoned_num, replace=False)
        poisoned_index = np.sort(poisoned_index)
        unique_labels = np.unique(temp_train_L, axis=0)
        for pi in poisoned_index:
            #label
            selected_row = np.random.choice(unique_labels.shape[0])
            label = unique_labels[selected_row]
            temp_train_L[pi] = label
            # image
            temp_image = np.copy(temp_train_x[pi])
            temp_train_x[pi] = backdoor.select(temp_image, temp_image.shape[-1],opt.backdoor_trigger)

        train_L = torch.from_numpy(temp_train_L)
        train_x = torch.from_numpy(temp_train_x)

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])
    return train_L,train_x,train_y,query_L,query_x,query_y,retrieval_L,retrieval_x,retrieval_y

def get_poisoned_test_dataset(X):
    query_x = X['query']
    retrieval_x = X['retrieval']
    print('...Deal Poisoned Dataset')
    poisoned_query_x = np.array(
        [backdoor.select(query_image, query_image.shape[-1],opt.backdoor_trigger) for query_image in tqdm(query_x)])
    poisoned_retrieval_x = np.array(
        [backdoor.select(retrieval_image, retrieval_image.shape[-1], opt.backdoor_trigger) for retrieval_image in tqdm(retrieval_x)])

    poisoned_query_x = torch.from_numpy(poisoned_query_x)
    poisoned_retrieval_x = torch.from_numpy(poisoned_retrieval_x)

    return poisoned_query_x,poisoned_retrieval_x

def load_pretrain_model(path):
    return sio.loadmat(path)


if __name__ == '__main__':
    a = {'s': [12, 33, 44],
         's': 0.111}
    import os
    with open('result.txt', 'w') as f:
        for k, v in a.items():
            f.write(k, v)