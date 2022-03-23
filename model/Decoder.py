import torch.nn as nn
import torch
import numpy as np

class Attention(nn.Module):

    def __init__(self, n_dim, n_heads):
        super(Attention, self).__init__()
        self.n_dim = n_dim
        self.n_heads = n_heads
        self.atten_size = self.n_dim // self.n_heads
        self.query = nn.Linear(self.atten_size, self.atten_size)
        self.key = nn.Linear(self.atten_size, self.atten_size)
        self.value = nn.Linear(self.atten_size, self.atten_size)
        self.fc = nn.Linear(self.n_dim, self.n_dim)

    def forward(self, query, key, value, mask):

        batch_size, query_words, _ = query.shape
        key_words = value_words = key.shape[1]
        
        q = self.query(query.reshape(batch_size, query_words, self.n_heads, self.atten_size))
        k = self.key(key.reshape(batch_size, key_words, self.n_heads, self.atten_size))
        v = self.value(value.reshape(batch_size, value_words, self.n_heads, self.atten_size))
        
        # (batch_size, query_words, n_heads, atten_size), (batch_size, key_words, n_heads, atten_size) -> (batch_size, n_heads, query_words, key_words)
        energy = torch.einsum('bnha, bmha -> bhnm', [q, k])/np.sqrt(self.atten_size)
    
        if mask != None:
            energy = energy.masked_fill(mask==0, float('-1e20'))

        dist = torch.softmax(energy, dim=3)

        # (batch_size, query_words, query_words, key_words), (batch_size, value_words, n_heads, atten_size) -> (batch_size, n_heads, query_words, atten_size)
        attention = torch.einsum('bhnm, bmha->bnha',[dist, v]).reshape(batch_size, query_words, self.n_dim)
        
        out = self.fc(attention)
        return out

class DecoderBlock(nn.Module):

    def __init__(self, n_dim, n_heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.masked_attention = Attention(n_dim, n_heads)
        self.encoder_decoder_attention = Attention(n_dim, n_heads)

        # Forward expansion
        self.fc = nn.Sequential(
            nn.Linear(n_dim, forward_expansion),
            nn.ReLU(),
            nn.Linear(forward_expansion, n_dim)
        )

        self.ln = [nn.LayerNorm(n_dim, n_dim, device=device) for _ in range(3)]
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, src_mask, trg_mask, return_extra=False):

        fst = self.dropout(self.ln[0](self.masked_attention(query, query, query, trg_mask) + query))
        snd = self.dropout(self.ln[1](self.encoder_decoder_attention(fst, key, value, src_mask) + fst))
        out = self.dropout(self.ln[2](self.fc(snd) + snd))

        if return_extra:
            # Only return extra in the last decoderblock
            return out, snd
        else:
            return out

class Decoder(nn.Module):

    def __init__(self, embedding, n_heads, forward_expansion, dropout, n_decoders, device):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.n_vocab = self.embedding.word_embeddings.num_embeddings
        self.n_dim = self.embedding.word_embeddings.embedding_dim

        # n - 1 regular decoderblock and 1 extra decoderblock returns extra
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(self.n_dim, n_heads, forward_expansion, dropout, device) for _ in range(n_decoders - 1)
            ]
        )
        self.extra_decoder = DecoderBlock(self.n_dim, n_heads, forward_expansion, dropout, device)
        self.fc_out = nn.Linear(self.n_dim, self.n_vocab)
    
    def forward(self, input_ids, token_type_ids, key, value, src_mask, trg_mask, return_extra=False):

        x = self.embedding(input_ids, token_type_ids)
        for decoder in self.decoders:
            x = decoder(x, key, value, src_mask, trg_mask)
        
        # Return the last two terms
        s_t, q_t = self.extra_decoder(x, key, value, src_mask, trg_mask, True)
        x = self.fc_out(s_t)
        if return_extra:
            # Only return extra when using pointer
            return x, s_t, q_t
        else:
            return x