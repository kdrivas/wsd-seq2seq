import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import re 
import numpy as np
from tqdm import tqdm, tqdm_notebook
from data import variable_from_sentence

class Beam():
    def __init__(self, decoder_input, decoder_context, decoder_hidden,
                    decoded_words=[], decoder_attentions=[], sequence_log_probs=[], decoded_index=[], decoder_arr_hidden=[], decoded_arr_output=[]):
        self.decoded_words = decoded_words
        self.decoded_index = decoded_index
        self.decoded_arr_output = decoded_arr_output
        self.decoder_attentions = decoder_attentions
        self.sequence_log_probs = sequence_log_probs
        self.decoder_input = decoder_input
        self.decoder_context = decoder_context
        self.decoder_hidden = decoder_hidden
        self.decoder_arr_hidden = decoder_arr_hidden

class Evaluator():
    def __init__(self, encoder, decoder, input_lang, output_lang, max_length, USE_CUDA):
        self.encoder = encoder
        self.decoder = decoder
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_length = max_length
        self.USE_CUDA = USE_CUDA
        
    def evaluate(self, sentence, k_beams):
        
        input_variable = variable_from_sentence(self.input_lang, sentence, USE_CUDA=self.USE_CUDA)
        input_length = input_variable.shape[0]
        
        encoder_hidden = self.encoder.init_hidden(1)
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)
        decoder_input = Variable(torch.LongTensor([[self.output_lang.vocab.stoi['<sos>']]]))

        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden
        
        if self.USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()
            
        decoded_words = []
        decoder_attentions = torch.zeros(self.max_length, self.max_length)

        beams = [Beam(decoder_input, decoder_context, decoder_hidden)]
        top_beams = []
        
        # Use decoder output as inputs
        for di in range(self.max_length):      
            new_beams = []
            for beam in beams:
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(
                    beam.decoder_input, beam.decoder_context, beam.decoder_hidden, encoder_outputs)     

                # Beam search, take the top k with highest probability
                topv, topi = decoder_output.data.topk(k_beams)

                for ni, vi in zip(topi[0], topv[0]):
                    new_beam = Beam(None, decoder_context, decoder_hidden, 
                                    beam.decoded_words[:], beam.decoder_attentions[:], beam.sequence_log_probs[:])
                    new_beam.decoder_attentions.append(decoder_attention.squeeze().cpu().data)
                    new_beam.sequence_log_probs.append(vi)

                    if ni == self.output_lang.vocab.stoi['<eos>'] or ni == self.output_lang.vocab.stoi['<pad>']: 
                        new_beam.decoded_words.append('<eos>')
                        top_beams.append(new_beam)

                    else:
                        new_beam.decoded_words.append(self.output_lang.vocab.itos[ni])                        
                        decoder_input = Variable(torch.LongTensor([[ni]]))
                        if self.USE_CUDA: decoder_input = decoder_input.cuda()
                        new_beam.decoder_input = decoder_input                        
                        new_beams.append(new_beam)                    
            
            new_beams = {beam: np.mean(beam.sequence_log_probs) for beam in new_beams}
            beams = sorted(new_beams, key=new_beams.get, reverse=True)[:k_beams]

            if len(beams) == 0:
                break
        
        if len(top_beams):
            top_beams = {beam: np.mean(beam.sequence_log_probs) for beam in top_beams}
        else:
            top_beams = {beam: np.mean(beam.sequence_log_probs) for beam in beams}

        # for beam in top_beams:
        #     print(beam.decoded_words, top_beams[beam])

        top_beams = sorted(top_beams, key=top_beams.get, reverse=True)[:k_beams]        

        decoded_words = top_beams[0].decoded_words

        return decoded_words
    
    def evaluate_sentence(self, sentence, k_beams=3):        
        output_words, decoder_attn, beams = self.evaluate(sentence, k_beams)
        output_sentence = ' '.join(output_words)
        
        print('>', sentence)
        print('<', output_sentence)
        print('')

    def evaluate_randomly(self, pairs, k_beams=3):
        pair = random.choice(pairs)
        print(pair)
        
        output_words, decoder_attn, beams = self.evaluate(pair[0], k_beams)
        output_sentence = ' '.join(output_words)
        
        print('>', pair[0])
        print('=', pair[1])
        print('<', output_sentence)
        print('')

    def evaluate_acc(self, pairs, k_beams=3, verbose=False):
        hint = 0
        for ix, pair in enumerate(pairs):
            temp = pair.split('\t')
            sentence, pos, id_senses = temp[0], int(temp[1]), temp[2:]
            
            output_words = self.evaluate(sentence, k_beams)
            torch.cuda.empty_cache()

            if len(output_words) > pos and output_words[pos] in id_senses:
                hint += 1
                            
            if verbose:
                print("----- ID")
                print(ix)
                print("----- tokens input")
                print(sentence)
                print("----- output predecido")
                print(' '.join(output_words))
                print("----- respuesta")
                print(id_senses)
                print()
                print("--- hints:  {}   --- acc: {}".format(hint, (hint / len(pairs))))
                
        return hint * 1.0 / len(pairs)
        
    def show_attention(self, input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
        plt.close()

    def evaluate_and_show_attention(self, input_sentence, k_beams=3):
        output_words, attentions, beams = self.evaluate(input_sentence, k_beams)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        self.show_attention(input_sentence, output_words, attentions)

    def evaluate_randomly_and_show_attention(self, pairs, k_beams=3):
        pair = random.choice(pairs)
        print(pair)        
        self.evaluate_and_show_attention(pair[0], k_beams)

    def get_candidates_and_references(self, pairs, arr_dep_test, k_beams=3):
        candidates = [self.evaluate(ix, pairs, arr_dep_test, k_beams)[0] for ix, pair in tqdm_notebook(enumerate(pairs))]
        candidates = [' '.join(candidate[:-1]) for candidate in candidates]
        references = [pair[1] for pair in pairs]
        return candidates, references