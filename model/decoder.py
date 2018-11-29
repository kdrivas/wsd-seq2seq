import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter

######################### ATTENTION ###########################

class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.
    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}
    Args:
        dim(int): The number of expected features in the output
    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    Examples::
         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)
    """
    def __init__(self, dim, USE_CUDA):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        output = output.transpose(1, 0)
        context = context.transpose(1, 0)
        
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        output = output.transpose(1, 0)
        return output, attn
            
######################### DECODER  LUONG ###########################

class Decoder_luong(nn.Module):
    def __init__(self, attn_method, hidden_size, output_size, emb_size, n_layers=1, dropout=0.1, lang=None, USE_CUDA=False):
        
        super(Decoder_luong, self).__init__()
        
        self.attn_method = attn_method
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.USE_CUDA = USE_CUDA
        self.lang = lang
        
        self.embedding = nn.Embedding(output_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, n_layers, dropout=dropout)
        self.attn = Attention(hidden_size, USE_CUDA)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.init_weights()
        
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        '''
        word_input: (seq_len, BS)
        last_context: (BS, encoder_hidden_size)
        last_hidden: (n_layers, BS, hidden_size)
        last_cell: (n_layers, BS, hidden_size)
        encoder_outputs: (seq_len, BS, encoder_hidden)
        < output: (BS, output_size)
        < attn_weights: (BS, 1, seq_len)
        '''
        #word_input = word_input.t()
        #encoder_outputs = encoder_outputs.transpose(1, 0)
        embedded = self.embedding(word_input)
        rnn_output, hidden = self.rnn(embedded, last_hidden)
        output, attn = self.attn(rnn_output, encoder_outputs)
        output = F.log_softmax(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(word_input.shape[1], self.output_size)
        
        return output, last_context, hidden, attn
    
    def init_weights(self):
        if self.lang:
            self.embedding.weight.data.copy_(self.lang.vocab.vectors)
            self.embedding.weight.requires_grad = False
            
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
                
        self.out.bias.data.fill_(0)
        self.out.weight.data.uniform_(-0.1, 0.1)
        
######################### DECODER  BAHDANAU ###########################

class Decoder_bahdanau(nn.Module):
    def __init__(self, attn_method, hidden_size, output_size, emb_size=None, n_layers=1, dropout=0.1, lang=None, USE_CUDA=False):
        super(Decoder_bahdanau, self).__init__()
        
        self.attn_method = attn_method
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.USE_CUDA = USE_CUDA
        self.lang = lang
        
        # (size of dictionary of embeddings, size of embedding vector)
        self.embedding = nn.Embedding(output_size, emb_size)
        self.rnn = nn.LSTM(emb_size + hidden_size, hidden_size, n_layers, dropout=dropout)
        # (input_features: embedding_size + encoder_hidden_size, output_features)
        self.out = nn.Linear(hidden_size * 2, output_size)        
        self.attn = Global_attn(attn_method, hidden_size, USE_CUDA)
        
        self.init_weights()
        
    def forward(self, word_input, last_hidden, last_cell, encoder_outputs):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input)#.view(1, 1, -1) # S=1 x B x N
        # word_embedded : (batch, emb_size)
        
        # Calculate attention weights and apply to encoder outputs
        # encoder_outputs : (seq_len, batch_size, hidden_size)
        # last_hidden : (num_layers, batch_size, hidden_size)
        # attn_weights : (batch_size, 1, seq_len)
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        
        # context : (batch_size, 1, hidden_size)
        
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded.unsqueeze(1), context), 2).permute(1,0,2)
        output, (hidden, cell) = self.rnn(rnn_input, (last_hidden, last_cell))
        
        # Final output layer
        output = output.squeeze(0) # B x N
        output = F.log_softmax(self.out(torch.cat((output, context.squeeze(1)), 1)), 1)
        # print('output sum: ', torch.sum(output.squeeze(), 1))
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, cell, attn_weights
    
    def init_weights(self):
        if self.lang:
            self.embedding.weight.data.copy_(self.lang.vocab.vectors)
            self.embedding.weight.requires_grad = False
            
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.out.bias.data.fill_(0)
        self.out.weight.data.uniform_(-0.1, 0.1)