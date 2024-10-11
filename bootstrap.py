import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
df = pd.read_csv('../../High_entropy_alloys/xgboost/data_final.csv')
#均匀抽样100组数据
df_sample = df.sample(n=100)
#bootstrap方法
#Cu,Mg,Mn,Fe,Si,Zn,Zr,temperature,strength
#基本神经网络
input_features=['Cu','Mg','Mn','Fe','Si','Zn','Zr','temperature']
X=df_sample[input_features].values
Y=df_sample['strength'].values

#试图对 pandas 的 DataFrame 对象调用 torch.tensor()，而 torch.tensor() 期望的是 list、numpy array 等类型的数据。因此，PyTorch 无法自动推断 DataFrame 的形状。#
X_tensor=torch.tensor(X,dtype=torch.float32).view(-1,8)
Y_tensor=torch.tensor(Y,dtype=torch.float32).view(-1,1)  #view(-1, 1) 将把 y_tensor 变成一个列向量，即具有 n 行和 1 列的张量，其中 n 是原来 y 中的数据长度。

#自定义数据集
#__init__: 初始化时，接收特征和标签数据。你可以传递 DataFrame 或者 NumPy 数组。
#__len__: 返回数据集的样本数量，PyTorch 用它来知道数据的大小。
#__getitem__: 根据索引 item 返回一对 (feature, label)。这是 PyTorch 进行数据索引的标准接口。
class CustomDataset(Dataset):
    def __init__(self,features,labels):
        self.features=features
        self.values=labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item],self.values[item]

dataset=CustomDataset(X_tensor,Y_tensor)

#划分训练集和测试集
train_dataset,test_dataset=train_test_split(dataset,test_size=0.2)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True)
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1=nn.Linear(8,16)
        self.fc2=nn.Linear(16,32)
        self.fc3=nn.Linear(32,1)
        self.relu=nn.ReLU()
    def forward(self,x):
        out=self.fc1(x)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.fc3(out)
        return out

model=MLP()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
epochs=60
for epoch in range(epochs):
    for i,(features,labels) in enumerate(train_loader):
        output=model(features)
        loss=criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1)%10==0:
        print(f"epoch:{epoch+1},loss:{loss.item():4f}")

model.eval()
with torch.no_grad():
    for i,(features,labels) in enumerate(test_loader):
        output=model(features)
        print(f"output:{output},labels:{labels}")
        loss=criterion(output,labels)
        print(f"loss:{loss.item():4f}")


#测试定义的数据集当中的方法
#a=len(dataset)
#b=dataset[0]

#output_feature=['strength']
print(df_sample)
