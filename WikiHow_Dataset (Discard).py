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
        with open(os.path.join(self.file_path, file_name),encoding='utf-8') as f:
            tmp = f.readlines()
            try:
                data = tmp[0][:-1]
                target = tmp[1][:-1]
            except Exception:
                print(target)

        return data, target

if __name__ == '__main__':
    
    import time

    dataset = WikiHow_Dataset('train')
    WikiData = DataLoader(dataset, batch_size = 1000, shuffle = True)

    start = time.time()

    for i, (data, target) in enumerate(WikiData):
        pass

    end = time.time()

    print(f'One epoch is done in {end - start:.2f} seconds')