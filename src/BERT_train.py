from cmath import inf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import torch.optim as optim
import torch
class WikiHow_Dataset(Dataset):
    def __init__(self, file_path):

        self.file = pd.read_csv(file_path)
        self.file["headline"] = self.file["headline"].str[1:]
        self.file["text"] = self.file["text"].str.replace("\n"," ",regex=False).str.strip().str.replace('\,','', regex=True).str.replace(r' +', ' ', regex=True)
        self.file = self.file.to_numpy()
        self.length = self.file.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):        
        
        x = self.file[idx,0][:-1]
        y = self.file[idx,1][:-1]

        return x, y
        
def BERT_train(model):

    dataset = WikiHow_Dataset('train.csv')
    WikiData = DataLoader(dataset, batch_size = 1000, shuffle = True)
    optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_loss=inf
    for epoch in range(5):
        
        train_epoch_loss=0

        for i, (data, target) in enumerate(WikiData):
            optimizer.zero_grad()
            output=model(data)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            train_epoch_loss=epoch_loss+loss
            #print(train_epoch_loss)
            train_epoch_loss=0
        


        dataset = WikiHow_Dataset('validate.csv')
        WikiData_validate = DataLoader(dataset, batch_size = 1000, shuffle = True)
        dataiter=iter(WikiData_validate)
        data_vali,target_vali=dataiter.next()
        output_texts=model(data_vali)
        
        #1000 is batch size
        #for z in range(1000):
            #这里不太清楚该怎么对比出valid loss

        if valid_loss<epoch_loss:
            torch.save(model.state_dict(),'bert_trained.pt')
            epoch_loss=valid_loss
    torch.save(model.state_dict(),'bert_trained.pt')
            

