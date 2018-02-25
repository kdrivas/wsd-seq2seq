import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from Preprocessing import SOS_token

######################### ENCODER ###########################

class Encoder_rnn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.1, USE_CUDA=False):
        super(Encoder_rnn, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.USE_CUDA = USE_CUDA
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True, batch_first=False)
        
    def forward(self, input_seqs, input_lengths, hidden = None, cell = None):
        embedded = self.embedding(input_seqs)

        self.lstm.flatten_parameters() 
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden, cell
    
    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size))
        if self.USE_CUDA: hidden = hidden.cuda()
        return hidden
    
    def init_cell(self, batch_size):
        cell = Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size))
        if self.USE_CUDA: cell = cell.cuda()
        return cell

######################### ATTENTION ###########################

class Global_attn(nn.Module):
    def __init__(self, method, hidden_size, USE_CUDA=False):
        super(Global_attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.USE_CUDA = USE_CUDA
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if self.USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy =torch.dot(hidden.view(-1), encoder_output.view(-1))
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = energy.view(-1,1)
            energy = hidden.mm(energy)
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.dot(self.v.view(-1), energy.view(-1))
        return energy
    
######################### DECODER ###########################

class Attn_decoder_rnn(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1, USE_CUDA=False):
        super(Attn_decoder_rnn, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.USE_CUDA = USE_CUDA

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Global_attn(attn_model, hidden_size, USE_CUDA)

    def forward(self, input_seq, last_hidden, last_cell, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N
    
        # Get current hidden state from input word and last hidden state
        self.lstm.flatten_parameters()
        rnn_output, (hidden, cell) = self.lstm(embedded, (last_hidden, last_cell))

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, cell, attn_weights

########################## DATA PARALLEL ###########################
    
class Disamb(nn.Module):
    def __init__(self, encoder, decoder, batch_size, n_layers, USE_CUDA=False):
        super(Disamb, self).__init__()
        
        self.small_hidden_size = 8
        self.small_n_layers = 2
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.n_layers = n_layers
        
        self.USE_CUDA = USE_CUDA

    def forward(self, input_lang, output_lang, input_batches, input_lengths, target_batches, target_lengths, tf_ratio, train):
        
        hidden_init = self.encoder.init_hidden(self.batch_size)
        cell_init = self.encoder.init_cell(self.batch_size)
        
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(input_batches, input_lengths, hidden_init, cell_init)

        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        all_decoder_outputs = Variable(torch.zeros(target_batches.data.size()[0], self.batch_size, output_lang.n_words))

        if self.USE_CUDA:
            all_decoder_outputs = all_decoder_outputs.cuda()
            decoder_input = decoder_input.cuda()
        
        for t in range(target_batches.data.size()[0]):
            decoder_output, decoder_hidden, decoder_cell, decoder_attn = self.decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs
            )
            all_decoder_outputs[t] = decoder_output # Store this step's outputs
            decoder_input = target_batches[t] # Next input is current target
            
            use_tf = random.random() < tf_ratio
            if use_tf and train:
                # De la data
                decoder_input = target_batches[t]
            else:
                # Del modelo
                topv, topi = decoder_output.data.topk(1)            
                decoder_input = Variable(topi.squeeze())
                
                if self.USE_CUDA: decoder_input = decoder_input.cuda()
        
        del decoder_output
        del decoder_hidden
        del decoder_attn
        
        return all_decoder_outputs, target_batches    

########################## TRAINING PARALLEL ###########################
   
def train_parallel(input_lang, output_lang, input_batches, input_lengths, target_batches, target_lengths, batch_size, disamb, disamb_optimizer, criterion, tf_ratio, max_length, clip=None, train=True, USE_CUDA=False):

    # Zero gradients of both optimizers
    disamb_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    all_decoder_outputs, target_batches = disamb(input_lang, output_lang, input_batches, input_lengths, target_batches, target_lengths, use_tf, train)
    
    # Loss calculation and backpropagation
    log_probs = F.log_softmax(all_decoder_outputs.view(-1, output_lang.n_words), dim=1)
    loss = criterion(log_probs, target_batches.view(-1))
    
    if train:
        loss.backward()
        torch.nn.utils.clip_grad_norm(disamb.parameters(), clip)
        disamb_optimizer.step()
    
    del all_decoder_outputs
    del target_batches
    
    return loss.data[0]
 
########################## TRAINING ###########################

def pass_batch(input_lang, output_lang, encoder, decoder, batch_size, input_batches, input_lengths, target_batches, target_lengths, tf_ratio, train=True, USE_CUDA=False):
        
    cell = encoder.init_cell(batch_size)
    hidden = encoder.init_hidden(batch_size)
        
    encoder_outputs, encoder_hidden, encoder_cell = encoder(input_batches, input_lengths, hidden, cell)

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    #decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size)) 
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    all_decoder_outputs = Variable(torch.zeros(target_batches.data.size()[0], batch_size, output_lang.n_words))

    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()
        decoder_input = decoder_input.cuda()
        #decoder_context = decoder_context.cuda()
        
    for t in range(target_batches.data.size()[0]):
        decoder_output, decoder_hidden, decoder_cell, decoder_attn = decoder(
            decoder_input, decoder_hidden, decoder_cell, encoder_outputs
        )
        all_decoder_outputs[t] = decoder_output # Store this step's outputs
        decoder_input = target_batches[t] # Next input is current target
        
        use_tf = random.random() < tf_ratio
        if use_tf and train:
            # De la data
            decoder_input = target_batches[t]
        else:
            # Del modelo
            topv, topi = decoder_output.data.topk(1) 
            decoder_input = Variable(topi.squeeze())
            
            if USE_CUDA: decoder_input = decoder_input.cuda()
        
    del decoder_output
    del decoder_hidden
    del decoder_attn
        
    return all_decoder_outputs, target_batches

def train(input_lang, output_lang, input_batches, input_lengths, target_batches, target_lengths, batch_size, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, use_tf, max_length, clip=None, train=True, USE_CUDA=False):
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    all_decoder_outputs, target_batches = pass_batch(input_lang, output_lang, encoder, decoder, batch_size, input_batches, input_lengths, target_batches, target_lengths, use_tf, train, USE_CUDA)
    
    # Loss calculation and backpropagation
    log_probs = F.log_softmax(all_decoder_outputs.view(-1, decoder.output_size), dim=1)
    loss = criterion(log_probs, target_batches.view(-1))
    
    if train:
        loss.backward()
        torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
        encoder_optimizer.step()
        decoder_optimizer.step()
    
    del all_decoder_outputs
    del target_batches
    
    return loss.data[0]