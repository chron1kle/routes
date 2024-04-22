import torch
import torch.nn as nn
from basic_functions import *
 
class KMeans(nn.Module):
    def __init__(self, n_clusters, n_features):
        super(KMeans, self).__init__()
        self.n_clusters = n_clusters
        self.centers = nn.Parameter(torch.randn(n_clusters, n_features))
 
    def forward(self, x, max_iter=100):
        for _ in range(max_iter):
            # 计算每个样本到聚类中心的距离
            distances = torch.cdist(x, self.centers)
 
            # 找到每个样本距离最近的聚类中心
            _, labels = torch.min(distances, dim=1)
 
            # 更新聚类中心为每个簇内样本的均值
            for i in range(self.n_clusters):
                cluster_points = x[labels == i]
                if len(cluster_points) > 0:
                    with torch.no_grad():
                        self.centers[i] = torch.mean(cluster_points, dim=0)
 
        return labels
 

if __name__ == "__main__":

    seg_length = 3
    offset = 5
    # filename = f'data\\segsConfi_{seg_length}_{offset}.json'
    # 生成一些示例数据
    data = load_seg_data(seg_length, offset, flag="confi")
    
    # 创建 KMeans 模型
    kmeans = KMeans(n_clusters=3, n_features=2)
    
    # 运行 K-means 算法
    cluster_labels = kmeans(data)
    
    # 打印聚类结果
    print("Cluster Labels:", cluster_labels)