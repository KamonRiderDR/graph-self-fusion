'''
Descripttion:   TODO load in following graph dataset.  
version: 
Author: Rui Dong
Date: 2023-10-08 14:16:23
LastEditTime: 2023-10-27 20:55:56
'''

import random
from typing import Optional, Callable, List
from copy import copy
import os.path as osp
import shutil
from tqdm import tqdm
import urllib
import os
import numpy as np
from zipfile import *
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import degree
import torch_geometric.transforms as T


#? 参考张可代码
r"""
在pyg的TUDataset数据库中，有部分数据集e.g.IMDB_BINARY没有节点特征（dataest.data.x==None）
重写该类，为没有节点特征的数据集生成节点特征。
所有图中节点的max degree<1000 使用OneHotDegree编码节点特征 (N, maxDegree)
所有图中节点的max degree>=1000 使用归一化的degree (N,1)
"""
class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                    'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
                pre_filter: Optional[Callable] = None,
                use_node_attr: bool = False, use_edge_attr: bool = False,
                cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.sizes = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
            # if hasattr(self.data, "original_x"):# TODO
            #     self.data.original_x = self.data.original_x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.data.x is None:
            max_degree = 0
            degs = []
            for data in self:  # data 表示一个图的数据
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                feature_transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                feature_transform = NormalizedDegree(mean, std)
            if self.pre_transform is None:
                self.pre_transform = feature_transform
            else:
                self.pre_transform = T.Compose([feature_transform, self.pre_transform])

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]  # 所有图数据list

            # data_list = [self.pre_transform(data) for data in data_list]
            new_data_list = []
            for data in tqdm(data_list):
                new_data_list.append(self.pre_transform(data))
            data_list = new_data_list
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices, sizes), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


class DDDataset(object):
    #数据集下载链接
    url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/DD.zip"
    
    def __init__(self, data_root="data", train_size=0.8):
        self.data_root = data_root
        self.maybe_download() #下载 并解压
        sparse_adjacency, node_labels, graph_indicator, graph_labels = self.read_data()
        #把coo格式转换为csr 进行稀疏矩阵运算
        self.sparse_adjacency = sparse_adjacency.tocsr()
        self.node_labels = node_labels
        self.graph_indicator = graph_indicator
        self.graph_labels = graph_labels
        
        self.train_index, self.test_index = self.split_data(train_size)
        self.train_label = graph_labels[self.train_index] #得到训练集中所有图对应的类别标签
        self.test_label = graph_labels[self.test_index] #得到测试集中所有图对应的类别标签

    def split_data(self, train_size):
        unique_indicator = np.asarray(list(set(self.graph_indicator)))
        #随机划分训练集和测试集 得到各自对应的图索引   （一个图代表一条数据）
        train_index, test_index = train_test_split(unique_indicator,
                                                    train_size=train_size,
                                                    random_state=1234)
        return train_index, test_index
    
    def __getitem__(self, index):
        mask = self.graph_indicator == index  
        #得到图索引为index的图对应的所有节点(索引)
        graph_indicator = self.graph_indicator[mask]
        #每个节点对应的特征标签
        node_labels = self.node_labels[mask]
        #该图对应的类别标签
        graph_labels = self.graph_labels[index]
        #该图对应的邻接矩阵
        adjacency = self.sparse_adjacency[mask, :][:, mask]
        return adjacency, node_labels, graph_indicator, graph_labels
    
    def __len__(self):
        return len(self.graph_labels)
    
    def read_data(self):
        #解压后的路径
        data_dir = os.path.join(self.data_root, "DD")
        print("Loading DD_A.txt")
        #从txt文件中读取邻接表(每一行可以看作一个坐标，即邻接矩阵中非0值的位置)  包含所有图的节点
        adjacency_list = np.genfromtxt(os.path.join(data_dir, "DD_A.txt"),
                                        dtype=np.int64, delimiter=',') - 1
        print("Loading DD_node_labels.txt")
        #读取节点的特征标签（包含所有图） 每个节点代表一种氨基酸 氨基酸有20多种，所以每个节点会有一个类型标签 表示是哪一种氨基酸
        node_labels = np.genfromtxt(os.path.join(data_dir, "DD_node_labels.txt"), 
                                    dtype=np.int64) - 1
        print("Loading DD_graph_indicator.txt")
        #每个节点属于哪个图
        graph_indicator = np.genfromtxt(os.path.join(data_dir, "DD_graph_indicator.txt"), 
                                        dtype=np.int64) - 1
        print("Loading DD_graph_labels.txt")
        #每个图的标签 （2分类 0，1）
        graph_labels = np.genfromtxt(os.path.join(data_dir, "DD_graph_labels.txt"), 
                                        dtype=np.int64) - 1
        num_nodes = len(node_labels) #节点数 （包含所有图的节点）
        #通过邻接表生成邻接矩阵  （包含所有的图）稀疏存储节省内存（coo格式 只存储非0值的行索引、列索引和非0值）
        #coo格式无法进行稀疏矩阵运算
        sparse_adjacency = sp.coo_matrix((np.ones(len(adjacency_list)), 
                                        (adjacency_list[:, 0], adjacency_list[:, 1])),
                                        shape=(num_nodes, num_nodes), dtype=np.float32)
        print("Number of nodes: ", num_nodes)
        return sparse_adjacency, node_labels, graph_indicator, graph_labels
    
    def maybe_download(self):
        save_path = os.path.join(self.data_root)
        #本地不存在 则下载
        if not os.path.exists(save_path):
            self.download_data(self.url, save_path)
        #对数据集压缩包进行解压
        if not os.path.exists(os.path.join(self.data_root, "DD")):
            zipfilename = os.path.join(self.data_root, "DD.zip")
            with ZipFile(zipfilename, "r") as zipobj:
                zipobj.extractall(os.path.join(self.data_root))
                print("Extracting data from {}".format(zipfilename))
    
    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载"""
        print("Downloading data from {}".format(url))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #下载数据集压缩包 保存在本地
        data = urllib.request.urlopen(url)
        filename = "DD.zip"
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(data.read())
        return True
    
# dataset = DDDataset()