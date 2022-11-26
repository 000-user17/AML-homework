import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pandas import DataFrame
from pytorch_pretrained_bert import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random
import matplotlib.pyplot as plt   #jupyter要matplotlib.pyplot
import math
from tqdm import tqdm
import re

def data_init(datas, mode='train'):
    datas_size = datas.shape[0]
    print(datas_size)
    #for l in range(datas_size):
    #    if type(datas['description'][l]) == str:
    #        continue
    #    elif math.isnan(datas['description'][l]):
    #        datas.drop([l],axis=0,inplace=True)  #删除description为空的行
    #datas = datas.reset_index(drop=True)  #从新将从0开始顺序标注序号
    #datas_size = datas.shape[0] #删除后的数据量
    
    if mode=='train':
        labels = [] #保存atis train.csv中一共有多少类
        for line in range(datas_size):
            if datas['target'][line] not in labels:          #注意这里的csv文件的标签栏默认为 'label',如有不同需更改
                labels.append(datas['target'][line])
    
    '''将description外的列为nan值的替换为[UNK]或0,然后拼接'''
    h2 = []
    for l in range(datas_size):
        if type(datas['description'][l]) == str:
            a=1
        elif math.isnan(datas['description'][l]):
            datas['description'][l] = ''
            
        if type(datas['neighbourhood'][l]) == str:
            a=1
        elif math.isnan(datas['neighbourhood'][l]):
            datas['neighbourhood'][l] = '[UNK]'
    
        if type(datas['latitude'][l]) == str:
            a=1
        elif math.isnan(datas['latitude'][l]):
            datas['latitude'][l] = 0
        
        if type(datas['longitude'][l]) == str:
            a=1
        elif math.isnan(datas['longitude'][l]):
            datas['longitude'][l] = 0
        
        if type(datas['type'][l]) == str:
            a=1
        elif math.isnan(datas['type'][l]):
            datas['type'][l] = '[UNK]'
    
        if type(datas['accommodates'][l]) == str:
            a=1
        elif math.isnan(datas['accommodates'][l]):
            datas['accommodates'][l] = 0
        
        if type(datas['bathrooms'][l]) == str:
            a=1
        elif math.isnan(datas['bathrooms'][l]):
            datas['bathrooms'][l] = '[UNK]'
        
        if type(datas['bedrooms'][l]) == str:
            a=1
        elif math.isnan(datas['bedrooms'][l]):
            datas['bedrooms'][l] = 0
    
        if type(datas['reviews'][l]) == str:
            a=1
        elif math.isnan(datas['reviews'][l]):
            datas['reviews'][l] = 0
    
        if type(datas['review_rating'][l]) == str:
            a=1
        elif math.isnan(datas['review_rating'][l]):
            datas['review_rating'][l] = 0
        
        if type(datas['review_scores_A'][l]) == str:
            a=1
        elif math.isnan(datas['review_scores_A'][l]):
            datas['review_scores_A'][l] = 0
    
        if type(datas['review_scores_B'][l]) == str:
            a=1
        elif math.isnan(datas['review_scores_B'][l]):
            datas['review_scores_B'][l] = 0
    
        if type(datas['review_scores_C'][l]) == str:
            a=1
        elif math.isnan(datas['review_scores_C'][l]):
            datas['review_scores_C'][l] = 0
        
        if type(datas['review_scores_D'][l]) == str:
            a=1
        elif math.isnan(datas['review_scores_D'][l]):
            datas['review_scores_D'][l] = 0
    
        if type(datas['instant_bookable'][l]) == str:
            a=1
        elif math.isnan(datas['instant_bookable'][l]):
            datas['instant_bookable'][l] = '[UNK]'
    
        s = str((datas['neighbourhood'][l])) +' '+ str(int(datas['review_rating'][l])) +' ' + str(int((datas['latitude'][l]+33)*(-100))) +' '+ str(int((datas['longitude'][l]-151)*(100))) +' '+ str(datas['type'][l]) +' '+ str(datas['accommodates'][l]) +' '+ str(datas['bathrooms'][l]) +' '+ str(int(datas['bedrooms'][l])) +' '+ str(int(datas['reviews'][l])) +' '+ str(int(datas['review_rating'][l])) +' '+ str(int(datas['review_scores_A'][l])) +' '+ str(int(datas['review_scores_B'][l]))+' '+ str(int(datas['review_scores_C'][l])) +' '+ str(int(datas['review_scores_D'][l])) +' '+ str(datas['instant_bookable'][l])
        h2.append(s)
    
    train_text = []#存放训练数据中未进行tokenize的text
    labels_idx = []
    for l in range(datas_size):
        h = datas['amenities'][l]  
        h = h.strip('[]')  
        h = re.sub('[",]', '', h)  
    
        train_text.append('[CLS] '+ h+ ' '+ h2[l] + ' ' + datas['description'][l]+ h +' [SEP]')
        if mode=='train':
            labels_idx.append(datas['target'][l]) 
        
    tokenizer = BertTokenizer.from_pretrained('./bert-pretrained') 
    train_data_tokens = [] 
    for l in range(datas_size):
        tokens = tokenizer.tokenize(train_text[l])
        train_data_tokens.append(tokens)
    
    max_len=0   
    for i in range(datas_size):
        max_len = max(max_len, len(train_data_tokens[i]))
    if max_len>300:
        max_len = 300  

    train_tokens_idx=[]
    for i in range(datas_size):
        token_idx = tokenizer.convert_tokens_to_ids(train_data_tokens[i])
        if len(token_idx) > max_len:
            token_idx = token_idx[0:max_len] 
        while len(token_idx) < max_len:
            token_idx.append(0)               
        train_tokens_idx.append(token_idx)
    
    if mode=='train':
        randnum = random.randint(0,100)
        random.seed(randnum)
        random.shuffle(train_tokens_idx)
        random.seed(randnum)
        random.shuffle(labels_idx)  #打乱训练集数据,这里注意必须打乱list类型的数据集，torch类型会导致重复
        print("打乱数据集完成")
    
    if mode=='train':
        tensor_datasets = TensorDataset(torch.tensor(train_tokens_idx), torch.tensor(labels_idx))
        train_tensor_datas = DataLoader(tensor_datasets, batch_size=256, shuffle=True, drop_last=True, num_workers=2)
    elif mode=='eval':
        tensor_datasets = TensorDataset(torch.tensor(train_tokens_idx))
        train_tensor_datas = DataLoader(tensor_datasets, batch_size=256, shuffle=False, drop_last=False, num_workers=2)
    
    return train_tensor_datas

'''定义模型类
参数
hidden_size: bert的embedding size
xlnet_hidden_dim:gru 隐藏层维度
xlnet_n_layers: gru层数
xlnet_bidirectional :gru是否双向
xlnet_dropout :gru dropout大小
num_classes:类数目
'''
class Bert_GRU(nn.Module):
    def __init__(self, hidden_size, xlnet_hidden_dim, xlnet_n_layers, xlnet_bidirectional, xlnet_dropout, num_classes):
        
        super(Bert_GRU,self).__init__()
        
        self.bert=BertModel.from_pretrained('./bert-pretrained')  
        for param in self.bert.parameters():
            param.requires_grad = False  
        
        self.rnn = nn.GRU(hidden_size,
                          xlnet_hidden_dim,
                          num_layers = xlnet_n_layers,
                          bidirectional = xlnet_bidirectional,
                          batch_first = True,
                          dropout = 0 if xlnet_n_layers < 2 else xlnet_dropout)
        
        self.fc1 = nn.Linear(xlnet_hidden_dim * 2 if xlnet_bidirectional else xlnet_hidden_dim, 512)
        self.fc = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(xlnet_dropout)
        
    def forward(self, tokens):
    
        with torch.no_grad():
            encoder_out,pooled = self.bert(tokens,output_all_encoded_layers=False) 
        _, hidden = self.rnn(encoder_out)
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        hidden = self.fc1(hidden)
        hidden = F.relu(hidden)
        output = self.fc(hidden)
        return output
    
'''BERT+CNN'''
class Bert_CNN(nn.Module):
    def __init__(self,num_filters, hidden_size, filter_size, dropout, num_classes):
        super(Bert_CNN,self).__init__()
        self.bert=BertModel.from_pretrained('./bert-pretrained')  
        for param in self.bert.parameters():
            param.requires_grad = False 
            
        self.convs=nn.ModuleList(

            [nn.Conv2d(1,num_filters,(k,hidden_size)) for k in filter_size]   
        )

        
        self.dropout=nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(num_filters*len(filter_size), 512)
        self.fc = nn.Linear(512, num_classes ) 

    def conv_and_pool(self, cnn_in, conv2d):
        cnn_in=conv2d(cnn_in)   
        cnn_in=F.relu(cnn_in)     
        cnn_in=cnn_in.squeeze(3)            
        cnn_in=F.max_pool1d(cnn_in, cnn_in.size(2))
      
        cnn_in = cnn_in.squeeze(2)  
  
        return cnn_in

    def forward(self, tokens):
  
        encoder_out,pooled = self.bert(tokens,output_all_encoded_layers=False) 
        cnn_in = encoder_out.unsqueeze(1)  
        cnn_out = torch.cat([self.conv_and_pool(cnn_in, conv2d) for conv2d in self.convs],1) 
        cnn_out = self.fc1(cnn_out)
        cnn_out = F.relu(cnn_out)
        cnn_out = self.dropout(cnn_out)
        out=self.fc(cnn_out) 
        
        return out

'''定义训练函数'''
def train(model, lossfunc, optimizer, epochs, tensor_datas): #增加了需要自己输入的epochs
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.train()

    model = model.to(device)
    losses = [] 
    accuracies = []
    iter = [] 

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    for epoch in range(epochs):
        loss_sum = 0
        accuracy=0
        for idx, datas in enumerate(tqdm(tensor_datas)):  
            tokens = datas[0].to(device)
            labels = datas[1].to(device)
        
            optimizer.zero_grad() 
            probs = model(tokens).squeeze()  
            probs.squeeze()
            
            loss = lossfunc(probs, labels) 
            loss_sum += loss.item()
            loss.backward()
            
            accuracy += (labels == torch.argmax(probs, dim=1)).sum()  #计算预测标签和真实标签相等的数量
            
            optimizer.step()
            scheduler.step()#学习率递减
        accuracy = accuracy / ((idx+1)*tensor_datas.batch_size)
        
        accuracies.append(accuracy.item()) 
        losses.append(loss_sum)
        iter.append(epoch)
        print("the loss of  training data "+ str(epoch) + "  " + str(loss_sum))
        print("the accuracy of training data   "+ str(epoch) + "  " + str(accuracy))
    
    plt.figure(1)
    plt.title("Losses")
    plt.xlabel("loss per epoch")
    plt.ylabel("Loss")
    plt.plot(iter, losses)

    plt.figure(2)
    plt.title("accuracies")
    plt.xlabel("ccuracy per epoch")
    plt.ylabel("Accuracy")
    plt.plot(iter, accuracies)

    plt.show()
    return accuracies, losses
def eval(tensor_datas, model):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    accuracy=0
    model.eval()
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(tensor_datas)):  
            tokens = datas[0].to(device)
            labels = datas[1].to(device)
            probs = model(tokens).squeeze()
            probs = F.softmax(probs, dim=1)
            accuracy += (labels == torch.argmax(probs, dim=1)).sum()  #计算预测标签和真实标签相等的数量
    accuracy = accuracy / ((idx+1)*tensor_datas.batch_size)
    print(accuracy)
    return accuracy.item()

def test(tensor_datas, model):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    accuracy=0
    model.eval()
    labels_pred = torch.tensor([])
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(tensor_datas)):  
            tokens = datas[0].to(device)
            
            probs = model(tokens).squeeze()
            probs = F.softmax(probs, dim=1)
            labels_pred = torch.cat([labels_pred, torch.argmax(probs, dim=1).to('cpu')])  
    
    return labels_pred

if __name__ == '__main__':
    '''定义模型，损失函数，优化器'''
    model = Bert_GRU(768, 256, 2, True, 0.5, 6)
    optimizer = torch.optim.AdamW(model.parameters())
    lossfuc = nn.CrossEntropyLoss()

    model = Bert_CNN(256, 768, (2,3,4), 0.5,6)
    optimizer = torch.optim.Adam(model.parameters())
    lossfuc = nn.CrossEntropyLoss()

    train_datas = pd.read_csv('./data/aml/train.csv')
    train_datas = data_init(train_datas)

    accuracies, losses = train(model, lossfuc, optimizer, 7, train_datas)
    
    valid_datas = pd.read_csv('./data/aml/valid.csv')
    valid_datas = data_init(valid_datas)
    
    eval(valid_datas, model)
    
    '''将全部训练数据（没划分验证集的）训练选择的模型BERtCNN'''
    train_datas = pd.read_csv('./data/aml/train_origin.csv')
    train_datas = data_init(train_datas)
    
    accuracies, losses = train(model, lossfuc, optimizer, 5, train_datas)

    eval(train_datas, model)
    
    '''放入测试举数据到Dataloader中'''
    test_datas = pd.read_csv('./data/aml/test.csv')
    test_datas = data_init(test_datas, mode='eval')
    
    '''得到测试集的预测标签'''
    labels_pred = test(test_datas, model)
    
    labels_pred_np = labels_pred.numpy()
    labels_pred_np
    
    '''预测结果放入txt文件'''
    import numpy as np
    np.savetxt('MG21330006.txt',labels_pred_np,fmt='%d')