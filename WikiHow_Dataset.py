from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

class WikiHow_Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_names = os.listdir(self.file_path)
        self.length = len(self.file_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):        
        
        file_name = self.file_names[idx]
        with open(os.path.join(self.file_path, file_name)) as f:
            tmp = f.readlines()
            data = tmp[0]
            target = tmp[2]

        return data, target

if __name__ == '__main__':
    dataset = WikiHow_Dataset('train')
    b = DataLoader(dataset, batch_size = 4, shuffle = True)
    for i,j in enumerate(b):
        print(i,j)