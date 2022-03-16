from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

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

if __name__ == '__main__':
    
    import time

    dataset = WikiHow_Dataset('test.csv')
    WikiData = DataLoader(dataset, batch_size = 1000, shuffle = True)

    start = time.time()

    for i, (data, target) in enumerate(WikiData):
        pass

    end = time.time()

    print(f'One epoch is done in {end - start:.2f} seconds')