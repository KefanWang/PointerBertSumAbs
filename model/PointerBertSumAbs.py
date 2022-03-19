import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

class PointerBertSumAbs(nn.Module):

    def __init__(self, pointer, BertModel, n_heads, forward_expansion, dropout, n_decoders):
        super(PointerBertSumAbs, self).__init__()
        self.pointer = pointer
        self.encoder = Encoder(BertModel)
        self.decoder = Decoder(
            self.encoder.BertModel.embeddings,
            n_heads,
            forward_expansion,
            dropout,
            n_decoders
        )
        self.cls_to_sig = nn.Linear(self.decoder.n_dim, 1)
        self.q_t_to_sig = nn.Linear(self.decoder.n_dim, 1)
        self.s_t_to_sig = nn.Linear(self.decoder.n_dim, 1)
        self.bert_to_context = nn.Linear(self.decoder.n_dim, self.decoder.n_vocab)
        self.q_t_to_context = nn.Linear(self.decoder.n_dim, self.decoder.n_vocab)
    
    def make_block_mask(self, trg):
        batch_size, num_words = trg.shape
        trg_mask = torch.tril(torch.ones((num_words, num_words))).expand(
            batch_size, 1, num_words, num_words
        )
        return trg_mask
    
    def forward(self, x_args, y_args):
        
        x_input_ids, x_token_type_ids, x_attention_mask = x_args['input_ids'], x_args['token_type_ids'], x_args['attention_mask']
        y_input_ids, y_token_type_ids, y_attention_mask = y_args['input_ids'], y_args['token_type_ids'], y_args['attention_mask']
        batch_size, num_words = y_input_ids.shape
        y_block_mask = self.make_block_mask(y_input_ids)
        y_padding_mask = y_attention_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, num_words, num_words)
        trg_mask = y_padding_mask * y_block_mask
        enc_output = self.encoder(x_input_ids, x_token_type_ids, x_attention_mask)
        all_output, CLS_output = enc_output[0], enc_output[1]
        if not self.pointer:
            return self.decoder(y_input_ids, y_token_type_ids, all_output, all_output, x_attention_mask.unsqueeze(1).unsqueeze(2), trg_mask)
        else:
            CLS_output_expand = CLS_output.unsqueeze(1).expand(batch_size, num_words, -1)
            out, s_t, q_t = self.decoder(y_input_ids, y_token_type_ids, all_output, all_output, x_attention_mask.unsqueeze(1).unsqueeze(2), trg_mask, True)
            p_gen = torch.sigmoid(self.cls_to_sig(CLS_output_expand) + self.q_t_to_sig(q_t) + self.s_t_to_sig(s_t))
            context = self.bert_to_context(all_output) + self.q_t_to_context(q_t)
            return p_gen * out + (1-p_gen) * context