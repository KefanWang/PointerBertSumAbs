import torch
import torch.nn as nn
from model.Encoder import Encoder
from model.Decoder import Decoder

class PointerBertSumAbs(nn.Module):

    def __init__(self, pointer, BertModel, n_heads, forward_expansion, dropout, n_decoders, device):
        super(PointerBertSumAbs, self).__init__()
        self.pointer = pointer
        self.encoder = Encoder(BertModel)
        self.decoder = Decoder(
            self.encoder.BertModel.embeddings,
            n_heads,
            forward_expansion,
            dropout,
            n_decoders,
            device
        )
        self.cls_to_sig = nn.Linear(self.decoder.n_dim, 1)
        self.q_t_to_sig = nn.Linear(self.decoder.n_dim, 1)
        self.s_t_to_sig = nn.Linear(self.decoder.n_dim, 1)
        self.bert_to_context = nn.Linear(self.decoder.n_dim, self.decoder.n_vocab)
        self.q_t_to_context = nn.Linear(self.decoder.n_dim, self.decoder.n_vocab)
        self.device = device
    
    def make_block_mask(self, trg):
        batch_size, num_words = trg.shape
        trg_mask = torch.tril(torch.ones((num_words, num_words))).expand(
            batch_size, 1, num_words, num_words
        )
        return trg_mask.to(self.device)
    
    def forward(self, x_input_ids, x_token_type_ids, x_attention_mask, y_input_ids, y_token_type_ids, y_attention_mask):
        
        x_input_ids = x_input_ids[:, :512]
        x_token_type_ids = x_token_type_ids[:, :512]
        x_attention_mask = x_attention_mask[:, :512]
        y_input_ids = y_input_ids[:, :512]
        y_token_type_ids = y_token_type_ids[:, :512]
        y_attention_mask = y_attention_mask[:, :512]

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
            mean_all_output = torch.mean(all_output,dim=1,keepdim=True).expand(batch_size, num_words, -1)
            context = self.bert_to_context(mean_all_output) + self.q_t_to_context(q_t)
            return p_gen * out + (1-p_gen) * context

    def predict(self, X, tokenizer, max_len=512):
        x_args= tokenizer(X,return_tensors='pt').to('cuda')
        x_input_ids, x_token_type_ids, x_attention_mask = x_args['input_ids'], x_args['token_type_ids'], x_args['attention_mask']
        batch_size = x_input_ids.shape[0]
        predictions = []
        for i in range(batch_size):
            x = x_input_ids[i].reshape(1, -1)
            x_token = x_token_type_ids[i].reshape(1, -1)
            x_mask = x_attention_mask[i].reshape(1, -1)
            counter = 1

            sentence = ['']
            while not sentence[0].endswith('[SEP]') and counter < max_len:
                
                y_args = tokenizer(sentence,return_tensors='pt').to('cuda')
                y_input_ids, y_token_type_ids, y_attention_mask = y_args['input_ids'], y_args['token_type_ids'], y_args['attention_mask']
                res = self.forward(x, x_token, x_mask, y_input_ids[:, :-1], y_token_type_ids[:, :-1], y_attention_mask[:, :-1])
                res = torch.softmax(res, dim=2).cpu().detach().numpy()[0, -1, :]
                rand = np.random.choice(np.arange(res.shape[0]), p=res)
                if sentence == ['']:
                    sentence = [tokenizer.decode(rand)]
                else:
                    sentence = [sentence[0] + ' ' + tokenizer.decode(rand)]
                counter += 1
            predictions.append(sentence)
        return predictions
