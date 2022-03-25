import torch
import torch.nn as nn
import numpy as np
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

        # Pointer
        self.cls_to_sig = nn.Linear(self.decoder.n_dim, 1)
        self.q_t_to_sig = nn.Linear(self.decoder.n_dim, 1)
        self.s_t_to_sig = nn.Linear(self.decoder.n_dim, 1)
        self.bert_to_context = nn.Linear(self.decoder.n_dim, self.decoder.n_vocab)
        self.q_t_to_context = nn.Linear(self.decoder.n_dim, self.decoder.n_vocab)

        self.device = device
           
    def make_block_mask(self, trg):

        # Make tasks for the decoder
        batch_size, num_words = trg.shape
        trg_mask = torch.tril(torch.ones((num_words, num_words))).expand(
            batch_size, 1, num_words, num_words
        )
        return trg_mask.to(self.device)
    
    def forward(self, x_input_ids, x_token_type_ids, x_attention_mask, y_input_ids, y_token_type_ids, y_attention_mask, in_origin):

        # Truncate inputs to map to the maximum allowed input size of Bert
        x_input_ids = x_input_ids[:, :512]
        x_token_type_ids = x_token_type_ids[:, :512]
        x_attention_mask = x_attention_mask[:, :512]
        y_input_ids = y_input_ids[:, :512]
        y_token_type_ids = y_token_type_ids[:, :512]
        y_attention_mask = y_attention_mask[:, :512]

        # Make mask for the decoder
        batch_size, num_words = y_input_ids.shape
        y_block_mask = self.make_block_mask(y_input_ids)
        y_padding_mask = y_attention_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, num_words, num_words)
        trg_mask = y_padding_mask * y_block_mask

        # Get the output from Bert
        enc_output = self.encoder(x_input_ids, x_token_type_ids, x_attention_mask)
        all_output, CLS_output = enc_output[0], enc_output[1]

        if not self.pointer:
            # Not use CLS_output when doing regular bertsumabs
            output = self.decoder(y_input_ids, y_token_type_ids, all_output, all_output, x_attention_mask.unsqueeze(1).unsqueeze(2), trg_mask)
        else:

            # Expand CLS_output to make the num_words match
            CLS_output_expand = CLS_output.unsqueeze(1).expand(batch_size, num_words, -1)

            # Get output, s_t, q_t
            out, s_t, q_t = self.decoder(y_input_ids, y_token_type_ids, all_output, all_output, x_attention_mask.unsqueeze(1).unsqueeze(2), trg_mask, True)

            # Calculate p_gen
            p_gen = torch.sigmoid(self.cls_to_sig(CLS_output_expand) + self.q_t_to_sig(q_t) + self.s_t_to_sig(s_t))

            # Average out all_output then expand to make the num_words match
            mean_all_output = torch.mean(all_output,dim=1,keepdim=True).expand(batch_size, num_words, -1)

            # Calculate the context vector
            context = self.bert_to_context(mean_all_output) + self.q_t_to_context(q_t)

            # Linear interpolate the output and context
            output = p_gen * out + (1-p_gen) * context
        
        if in_origin:
            # Generate text only in the original input
            for i in range(batch_size):
                temp = output[i, :, x_input_ids[i]]
                output[i, :, :] = -1e20
                output[i, :, x_input_ids[i]] = temp
                output[i, :, 101] = -1e20
                output[i, :, 0] = -1e20

        return output

    def predict(self, X, tokenizer, in_origin, most_prop, max_len=100):

        # Tokenize words
        x_args= tokenizer(X,return_tensors='pt',padding=True).to('cuda')
        x_input_ids, x_token_type_ids, x_attention_mask = x_args['input_ids'], x_args['token_type_ids'], x_args['attention_mask']

        # Get batch_size
        batch_size = x_input_ids.shape[0]
        possb = np.arange(self.decoder.n_vocab)

        predictions = []
        for i in range(batch_size):

            '''Currently iterate over all samples - more efficient ways remain to be explored'''

            # Get the ith sample and reshape to the (batch_size, num_words)
            x = x_input_ids[i].reshape(1, -1)
            x_token = x_token_type_ids[i].reshape(1, -1)
            x_mask = x_attention_mask[i].reshape(1, -1)

            # Use counter to track the number of words in the summary
            counter = 0
            
            # Initialize output to be empty string
            sentence = ['']

            # Stop producing the words when the sentence ends with '[SEP]' or exceeds the maximum allowed length
            while not sentence[0].endswith('[SEP]') and counter < max_len:
                
                # Tokenize the sentence
                y_args = tokenizer(sentence,return_tensors='pt',padding=True).to('cuda')
                y_input_ids, y_token_type_ids, y_attention_mask = y_args['input_ids'], y_args['token_type_ids'], y_args['attention_mask']

                # Get the output - predict without [SEP] generated by tokenizer
                res = self.forward(x, x_token, x_mask, y_input_ids[:, :-1], y_token_type_ids[:, :-1], y_attention_mask[:, :-1], in_origin)

                # Get the distribution of the last output - only predict the final word
                res = torch.softmax(res, dim=2).cpu().detach().numpy()[0, -1, :]

                # Random sample the target word

                if most_prop:
                    rand = np.argmax(res)
                else:
                    rand = np.random.choice(possb, p=res)
                word = tokenizer.decode(rand)

                if sentence == ['']:

                    # Replace to new output if in the initial state
                    if word.startswith('##'):
                        continue
                    sentence = [tokenizer.decode(rand)]
                else:

                    # Concat the original sentence with the new word

                    if word.startswith('##'):
                        sentence = [sentence[0] + word[2:]]
                    else:
                        sentence = [sentence[0] + ' ' + word]

                counter += 1
            
            if sentence[0].endswith('[SEP]'):
                predictions.append(sentence[0][:-6])
            else:
                predictions.append(sentence[0])
                
        return predictions
