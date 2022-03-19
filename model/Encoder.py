import torch.nn as nn
from transformers import AutoModel

class Encoder(nn.Module):

    def __init__(self, BertModel):
        super(Encoder, self).__init__()
        self.BertModel = AutoModel.from_pretrained(BertModel)

    def save_to_disk(self, file_name='bert-base-uncased'):
        self.BertModel.save_pretrained(file_name)

    def forward(self, input_ids, token_type_ids, attention_mask):
        out = self.BertModel(input_ids, token_type_ids, attention_mask)
        return out