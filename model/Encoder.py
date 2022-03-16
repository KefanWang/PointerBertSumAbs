from turtle import forward
import torch.nn as nn
from transformers import BertTokenizerFast, AutoModel
import os

class Encoder(nn.Module):

    def __init__(self, vocab_path=os.path.join('data','bert-base-uncased-vocab.txt'), BertModel='bert-base-uncased'):
        super(Encoder, self).__init__()
        self.BertModel = AutoModel.from_pretrained(BertModel)
        self.tokenizer = BertTokenizerFast(vocab_path)

    def save_to_disk(self, file_name='bert-base-uncased'):
        self.BertModel.save_pretrained(file_name)

    def forward(self, x):
        tokenizer_data = self.tokenizer(x, return_tensors='pt')
        out = self.BertModel(**tokenizer_data)
        return out

if __name__ == '__main__':

    encoder = Encoder()
    x = 'Hello World'
    print(encoder.forward(x))