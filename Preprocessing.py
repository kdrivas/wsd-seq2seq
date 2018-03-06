import unicodedata
import string
import re
import random
import time
import math
import os
import sys
import pandas as pd
import numpy as np

import nltk

import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from stanfordcorenlp import StanfordCoreNLP
from nltk.parse.stanford import StanfordParser
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

import enchant

import torchtext 
from torchtext import data
from torchtext import datasets

PAD_token = 1
SOS_token = 2
EOS_token = 3

# label of dependencies https://nlp.stanford.edu/pubs/USD_LREC14_paper_camera_ready.pdf

DEP_LABELS = ['ROOT', 'ACL','ACVLCL', 'ADVMOD', 'AMOD', 'APPOS', 'AUX', 'CASE', 'CC', 'CCOMP',
               'CLF', 'COMPOUND', 'CONJ', 'COP', 'CSUBJ', 'DEP', 'DET',
               'DISCOURSE', 'DISLOCATED', 'EXPL', 'FIXED', 'FLAT', 'GOESWITH',
               'IOBJ', 'LIST', 'MARK', 'NMOD', 'NSUBJ', 'NUMMOD',
               'OBJ', 'OBL', 'ORPHAN', 'PARATAXIS', 'PUNXT', 'REPARANDUM', 'VOCATIVE',
               'XCOMP']

_DEP_LABELS_DICT = {label:ix for ix, label in enumerate(DEP_LABELS)}

def find_words_cutoff(sentences, ner):
    words_cutoff = []
    
    for ix, sentence in enumerate(sentences):
        first_word = True
        words_cutoff.append("")
        tokens = sentence.split()
        tags = ner.tag(tokens)
        for tag in tags:
            # Veryfing if word is He, She, It, That or To
            if(tag[0] == ''):
                continue
            if(tag[1] == 'O' and \
               (tag[0][0] == 'S' or tag[0][0] == 'H' or tag[0][0] == 'I' or tag[0][0] == 'T')):
                # we want to get a sentence of the form of word1|word2|word3
                if(first_word):
                    first_word = False
                else:
                    words_cutoff[ix] += "|"
                    
                words_cutoff[ix] += tag[0]
                
    return words_cutoff

def is_target_tree(tree, word_target):

    for pos in tree.pos():
        if(pos[0] == word_target):
            return True
    
    return False

def prune_subtree(tree, word_target):
    subtrees = tree.subtrees(filter = lambda x: x.label()=="S" )
    
    for t in reversed(list(subtrees)):
        pos = t.treeposition()
        if(len(pos) != 0 and not is_target_tree(t, word_target)):
            del tree[pos]
            
    return tree

def find_subtree(tree, pos_word):
    aux = tree.copy(deep=True)
    index = -1
    
    for pos in range(len(pos_word) - 1):
        label = aux[pos_word[pos]].label()
        if(label == 'S'):
            index = pos
        aux = aux[pos_word[pos]]

    
    l_tree = list(tree[pos_word[:index+1]])
    present_np = False
    for child in range(len(l_tree)):
        if(l_tree[child].label() == 'NP'):
            present_np = True

    if(not present_np and index != -1) :      
        parent_tree = tree[pos_word[:index+1]].parent()
        pos_parent = parent_tree.treeposition()
        
        while(tree[pos_parent].label() != 'NP' and tree[pos_parent].label() != 'ROOT'):
            parent_tree = tree[pos_parent].parent()
            pos_parent = parent_tree.treeposition()
    else:
        pos_parent = pos_word[:index+1]

    return tree[pos_parent].copy(deep=True)

def find_last_pos(lst_pos):
    index = 0
    
    for pos in reversed(lst_pos):
        if(pos[1] != 'WDT' and pos[1] != ',' and pos[1] != 'IN' and pos[1] != 'WRB'):
            return index 
        index += 1
        
    return -1

def tree_to_sentence(tree):
    sentence = ""

    lst_pos = list(tree.pos())
    len_pos = len(lst_pos)
    last_pos = find_last_pos(lst_pos)
    first_comma = True
    
    for ix, pair in enumerate(tree.pos()):      
        if(ix >= len_pos - last_pos):
            break
        
        if(pair[1] != ',' and pair[1] != 'WRB' and pair[1] != 'IN'):
            first_comma = False
        
        if(not first_comma):
            sentence += pair[0] + ' '

    sentence += '.'
    
    return sentence

def sentence_prune(sentence, lst_index, parser):
    
    raw_tree = parser.raw_parse(sentence)
    aux = list(raw_tree)
    tree = nltk.ParentedTree.convert(aux[0])
    lst_sentences = []
    
    for index in lst_index:
        
        word_target = sentence.split()[index]
        pos_word = tree.leaf_treeposition(index)   
        aux_tree = find_subtree(tree, pos_word)
        aux_tree = prune_subtree(aux_tree, word_target)
        lst_sentences.append(tree_to_sentence(aux_tree))
 
    return list(set(lst_sentences))

def join_words(sentences, word_dict):
    arr_sentences = []
    
    for sentence in sentences:
        tokens = sentence.split()
        ant = ''
        new_sentence = ''
        add_word = True
        
        for ix, token in enumerate(tokens):
            if(add_word):
                if(token == '-'  and ix > 0 and ix < (len(tokens) - 1)):
                    join_word = tokens[ix-1] + '-' + tokens[ix+1]
                    if word_dict.check(join_word):
                        ant = join_word + ' '
                        add_word = False

                if(add_word):
                    new_sentence += ant
                    ant = token + " "
            else:
                add_word = True

        new_sentence += ant
        arr_sentences.append(new_sentence)
    
    return arr_sentences

def remove_unnecesary_char(sentence):
    sentence = sentence.strip(' ')
    sentence = sentence.lstrip(')')
    
    return sentence

def find_index(sentence):
    words = sentence.split()
    
    lst_index = []
    for ix_word, word in enumerate(words):
        if(word.startswith('<head>')):
            lst_index.append(ix_word)

    return lst_index

def find_trim_sentence(sentence, iz_del, der_del):
    cont_iz = 0
    seqs = []
    seq = ''
    tokens = sentence.split()
    add = False
    
    for token in tokens:
        if(token == iz_del):
            cont_iz += 1
        
        if(cont_iz > 0):
            add = True
            seq += token + ' '
        
        if(token == der_del):
            cont_iz -= 1
        
        if(cont_iz == 0 and add):
            seq = seq.strip(' ')
            seqs.append(seq)
            seq = ''
            add = False
            
    return seqs

def remove_LRB_RRB(sentences, iz_del, der_del):
    arr_sentence = []
    
    for sentence in sentences:
        seqs = find_trim_sentence(sentence, iz_del, der_del)
        for seq in seqs:
            arr = find_index(seq)
            if(len(arr) == 0):
                sentence = sentence.replace(seq, '')
            else:
                sentence = seq.strip(iz_del + der_del + ' ')
                break
                
        arr_sentence.append(sentence)
    
    return arr_sentence

def process_instance(ix_ins, text, ner, parser, word_dict, nlp, is_train = True, sense_ids = None, prune_sentence = False, verbose = False):
    pairs = []
    sentences = []
    
    if is_train:
        sense_ids = re.findall(r'senseid=\"(.*?)\"', text, re.DOTALL)
        
    context = re.findall(r'<context>(.*?)</context>', text, re.DOTALL)
    word_ambiguos = re.findall(r'<head>(.*?)</head>', context[0], re.DOTALL)
    
    c = re.split(r'[\.|:|?|!]', context[0])
    
    words_cutoff = find_words_cutoff(c, ner)
    
    for ix, sent in enumerate(c):
        if(len(words_cutoff[ix]) != 0):
            sentences.extend(re.split(r'\s(?=(?:' + words_cutoff[ix] + r')\b)', sent))
        else:
            sentences.append(sent)
    
    for sentence in sentences:
        if(sentence.endswith('and')):
            sentence = sentence.rsplit(' and', 1)[0]
            
        tags = re.findall(r'<head>(.*?)</head>', sentence)
        if(len(tags) != 0):
            
            sentence = remove_unnecesary_char(sentence)
            index_word = find_index(sentence)            
            
            if(verbose):
                print('---oracion')
                print(sentence) 

            if(prune_sentence):
                sentence = re.sub(r'<head>(.*?)</head>', word_ambiguos[0], sentence)
                sentences_prune = sentence_prune(sentence, index_word, parser)
                sentences_prune = remove_LRB_RRB(sentences_prune, '-LRB-', '-RRB-')
            else:
                sentences_prune = []
                sentences_prune.append(sentence)
                sentences_prune = remove_LRB_RRB(sentences_prune, '(', ')')
                
            
            if(verbose):
                print('---oracion sin parentesis')
                print(sentences_prune[0])
                print('\n')
                
            for s in sentences_prune:

                for sense_id in sense_ids:   
                    pair = [[],[],[],[]]
                    sense_id = re.sub(r'%|:', '', sense_id)
                    if(prune_sentence):
                        pair[0] = s
                        pair[1] = re.sub(word_ambiguos[0], word_ambiguos[0] + '_' + sense_id, s)
                        pair[2] = word_ambiguos[0] + '_' + sense_id
                        pair[3] = ix_ins
                    else:
                        pair[0] = re.sub(r'<head>(.*?)</head>', word_ambiguos[0], s)
                        pair[1] = re.sub(r'<head>(.*?)</head>', word_ambiguos[0] + '_' + sense_id, s)
                        pair[2] = word_ambiguos[0] + '_' + sense_id
                        pair[3] = ix_ins
                    pairs.append(pair)
        
    return pairs

def load_senses(path):
    
    senses_all = []
    with open(path, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            senses = []
            words = line.split()
            for ix, word in enumerate(words):
                if ix > 1:
                    senses.append(word)
                    
            senses_all.append(senses)
    
    return senses_all

def construct_pairs(path_source, path_model, is_train = True, test_path = None, prune_sentence = False, verbose=True):
    
    ner = StanfordNERTagger(path_model + 'stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz',
                        path_model + 'stanford-ner-2017-06-09/stanford-ner.jar',
                        encoding='utf-8')

    parser=StanfordParser(path_model + "stanford-parser-full-2017-06-09/stanford-parser.jar", \
                     path_model + "stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar")

    #nlp = StanfordCoreNLP(path_model + "stanford-corenlp-full-2018-02-27/")
    nlp = None
    
    word_dict = enchant.Dict('en_US')
    
    with open(path_source, 'r') as f:
        xml = f.read()

    if(not is_train):
        senses_all = load_senses(test_path)    
    
    instances = re.findall(r'<instance(.*?)</instance>', xml, re.DOTALL)
    pairs= []

    for ix_ins, instance in enumerate(instances):
        data = '<instance' + instance + '</instance>'
        data = re.sub(r'[^\x20-\x7E]', '', data)
        data = re.sub(r' n\'t', 'n\'t', data)
        data = re.sub(r'wou \'d', 'uld', data)

        data = re.sub(r' \'re', ' are', data)
        data = re.sub(r' \'ve', ' have', data)
        
        data = re.sub(r'it \'s', 'it is', data)
        data = re.sub(r'he \'s', 'he is', data)
        data = re.sub(r'i \'m', 'i am', data)
        data = re.sub(r'It \'s', 'it is', data)
        data = re.sub(r'He \'s', 'he is', data)
        data = re.sub(r'I \'m', 'i am', data)
        
        data = re.sub(r' \'d', 'd', data)
        data = re.sub(r'&', '', data)
        
        if(is_train):
            pairs.extend(process_instance(ix_ins, data, ner, parser, word_dict, nlp, is_train, None, prune_sentence, verbose))
        else:
            pairs.extend(process_instance(ix_ins, data, ner, parser, word_dict, nlp, is_train, senses_all[ix_ins], prune_sentence, verbose))
        
    
    return np.array(pairs)

###################### GET LANGUAGE MODEL DATA ############################

def process_instance_LM(text, verbose = False):
    pairs = []
        
    context = re.findall(r'<context>(.*?)</context>', text, re.DOTALL)
    word_ambiguos = re.findall(r'<head>(.*?)</head>', context[0], re.DOTALL)
    context = re.sub(r'<head>(.*?)</head>', word_ambiguos[0], context[0])
    
    c = re.split(r'[\.]', context)
        
    if verbose:
        print("------ sentences")
        print(c)
        print()

    return c

def construct_LM_data(path_source, verbose=True):
        
    with open(path_source, 'r') as f:
        xml = f.read() 
    
    instances = re.findall(r'<instance(.*?)</instance>', xml, re.DOTALL)
    pairs= []

    for ix_ins, instance in enumerate(instances):
        data = '<instance' + instance + '</instance>'
        data = data.lower()
        
        data = re.sub(r'[^\x20-\x7E]', '', data)
        data = re.sub(r' n\'t', 'nt', data)
        data = re.sub(r' \'re', ' are', data)
        data = re.sub(r' \'ve', ' have', data)
        data = re.sub(r'it \'s', 'it is', data)
        data = re.sub(r'he \'s', 'he is', data)
        data = re.sub(r'i \'m', 'i am', data)
        data = re.sub(r' \'d', 'd', data)
        data = re.sub(r'wou \'d', 'uld', data)
        
        pairs.extend(process_instance_LM(data, verbose))
            
    return np.array(pairs)

###################### BATCHES ############################

def get_all_id(pairs):
    id_pairs = []
    
    for i in range(len(pairs)):
        id_pairs.append(int(pairs[i][3]))
        
    return id_pairs

def pad_seq(lang, seq, max_length):
    seq += [lang.vocab.stoi["<pad>"] for i in range(max_length - len(seq))]
    return seq

def indexes_from_sentence(lang, sentence):
    return [lang.vocab.stoi[word] for word in sentence] + [lang.vocab.stoi["<eos>"]]

def random_batch(input_lang, output_lang, batch_size, pairs, return_dep_tree=False, arr_dep=None, USE_CUDA=False):
    input_seqs = []
    target_seqs = []
    id_pairs = []
    arr_aux = []
    
    id_arr = list(range(len(pairs)))
    for i in range(batch_size):
        id_random = random.choice(id_arr)
        pair = pairs[id_random]
        
        if arr_dep and return_dep_tree:
            arr_aux.append(arr_dep[id_random])
        elif return_dep_tree:
            arr_aux.append(pair[2])
        
        id_pairs.append(id_random)
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(output_lang, pair[1]))

    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(input_lang, s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(output_lang, s, max(target_lengths)) for s in target_seqs]

    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    
    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

#
#     nlp = StanfordCoreNLP(r'/home/krivas/projects/wsd-v2/stanford-corenlp-full-2018-01-31/')

#     sentence = 'which you step on to activate it'
#     de = nlp.dependency_parse(sentence)

#     arr_dep = []
#     arr_dep.append(de)
#     arr_dep.append(de)
#     arr_dep.append(de)
#     arr_dep.append(de)
    
    max_length = max(input_lengths)
    matrix_size = batch_size * max_length
    
    #Initialize adjancencies matrixes

    adj_arc_in = np.zeros((matrix_size, 2), dtype='int32')
    adj_lab_in = np.zeros(matrix_size, dtype='int32')

    adj_arc_out = np.zeros((matrix_size, 2), dtype='int32')
    adj_lab_out = np.zeros(matrix_size, dtype='int32')
    
    #Initialize mask matrix

    mask_in = np.zeros(matrix_size, dtype='float32')
    mask_out = np.zeros(matrix_size, dtype='float32')

    mask_loop = np.ones((matrix_size, 1), dtype='float32')
    
    # Enable dependency label batch
    if return_dep_tree:
        
        #Get adjacency matrix for incoming and outgoing arcs
        for idx_sentence, dep_sentence in enumerate(arr_aux):
            for idx_arc, arc in enumerate(dep_sentence):
                if(arc[0] != 'ROOT') and arc[0].upper() in DEP_LABELS:
                    #get index of words in the sentence
                    arc_1 = int(arc[1]) - 1
                    arc_2 = int(arc[2]) - 1

                    idx_in = (idx_arc) + idx_sentence * max_length
                    idx_out = (arc_2) + idx_sentence * max_length

                    #Make adjacency matrix for incoming arcs
                    adj_arc_in[idx_in] = np.array([idx_sentence, arc_2]) 
                    adj_lab_in[idx_in] = np.array([_DEP_LABELS_DICT[arc[0].upper()]]) 

                    #Setting mask to consider that index
                    mask_in[idx_in] = 1

                    #Make adjacency matrix for outgoing arcs
                    adj_arc_out[idx_out] = np.array([idx_sentence, arc_1])   
                    adj_lab_out[idx_out] = np.array([_DEP_LABELS_DICT[arc[0].upper()]])

                    #Setting mask to consider that index
                    mask_out[idx_out] = 1


    adj_arc_in = Variable(torch.LongTensor(np.transpose(adj_arc_in)))
    adj_arc_out = Variable(torch.LongTensor(np.transpose(adj_arc_out)))

    adj_lab_in = Variable(torch.LongTensor(adj_lab_in))
    adj_lab_out = Variable(torch.LongTensor(adj_lab_out))

    mask_in = Variable(torch.FloatTensor(mask_in.reshape((matrix_size, 1))))
    mask_out = Variable(torch.FloatTensor(mask_out.reshape((matrix_size, 1))))
    mask_loop = Variable(torch.FloatTensor(mask_loop))
    
    if USE_CUDA:
        adj_arc_in = adj_arc_in.cuda()
        adj_arc_out = adj_arc_out.cuda()
        adj_lab_in = adj_lab_in.cuda()
        adj_lab_out = adj_lab_out.cuda()
        
        mask_in = mask_in.cuda()
        mask_out = mask_out.cuda()
        mask_loop = mask_loop.cuda()
        
    return input_var, input_lengths, target_var, target_lengths,\
            adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop,\
id_pairs

def generate_batch(input_lang, output_lang, batch_size, pairs, pos_instance=None, return_dep_tree=False, arr_dep=None, USE_CUDA=False):
    input_seqs = []
    target_seqs = []
    id_pairs = []
    arr_aux = []
    
    id_arr = list(range(len(pairs)))
    for i in range(batch_size):
        if pos_instance:
            id_pair = pos_instance + i
        else:
            id_pair = random.choice(id_arr)
        
        if id_pair >= len(pairs):
            break
        
        pair = pairs[id_pair]
        
        if arr_dep and return_dep_tree:
            arr_aux.append(arr_dep[id_pair])
        elif return_dep_tree:
            arr_aux.append(pair[2])
        
        id_pairs.append(id_pair)
        input_seqs.append(pair[0].split())
        target_seqs.append(pair[1].split())
  
    input_lengths = [len(s) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    
    input_padded = [indexes_from_sentence(input_lang, seq) for seq in input_lang.pad(input_seqs)]
    target_padded = [indexes_from_sentence(output_lang, seq) for seq in output_lang.pad(target_seqs)]
    
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    
    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

#
#     nlp = StanfordCoreNLP(r'/home/krivas/projects/wsd-v2/stanford-corenlp-full-2018-01-31/')

#     sentence = 'which you step on to activate it'
#     de = nlp.dependency_parse(sentence)

#     arr_dep = []
#     arr_dep.append(de)
#     arr_dep.append(de)
#     arr_dep.append(de)
#     arr_dep.append(de)
    
    max_length = max(input_lengths)
    matrix_size = batch_size * max_length
    
    #Initialize adjancencies matrixes

    adj_arc_in = np.zeros((matrix_size, 2), dtype='int32')
    adj_lab_in = np.zeros(matrix_size, dtype='int32')

    adj_arc_out = np.zeros((matrix_size, 2), dtype='int32')
    adj_lab_out = np.zeros(matrix_size, dtype='int32')
    
    #Initialize mask matrix

    mask_in = np.zeros(matrix_size, dtype='float32')
    mask_out = np.zeros(matrix_size, dtype='float32')

    mask_loop = np.ones((matrix_size, 1), dtype='float32')
    
    # Enable dependency label batch
    if return_dep_tree:
        
        #Get adjacency matrix for incoming and outgoing arcs
        for idx_sentence, dep_sentence in enumerate(arr_aux):
            for idx_arc, arc in enumerate(dep_sentence):
                if(arc[0] != 'ROOT') and arc[0].upper() in DEP_LABELS:
                    #get index of words in the sentence
                    arc_1 = int(arc[1]) - 1
                    arc_2 = int(arc[2]) - 1

                    idx_in = (idx_arc) + idx_sentence * max_length
                    idx_out = (arc_2) + idx_sentence * max_length

                    #Make adjacency matrix for incoming arcs
                    adj_arc_in[idx_in] = np.array([idx_sentence, arc_2]) 
                    adj_lab_in[idx_in] = np.array([_DEP_LABELS_DICT[arc[0].upper()]]) 

                    #Setting mask to consider that index
                    mask_in[idx_in] = 1

                    #Make adjacency matrix for outgoing arcs
                    adj_arc_out[idx_out] = np.array([idx_sentence, arc_1])   
                    adj_lab_out[idx_out] = np.array([_DEP_LABELS_DICT[arc[0].upper()]])

                    #Setting mask to consider that index
                    mask_out[idx_out] = 1


    adj_arc_in = Variable(torch.LongTensor(np.transpose(adj_arc_in)))
    adj_arc_out = Variable(torch.LongTensor(np.transpose(adj_arc_out)))

    adj_lab_in = Variable(torch.LongTensor(adj_lab_in))
    adj_lab_out = Variable(torch.LongTensor(adj_lab_out))

    mask_in = Variable(torch.FloatTensor(mask_in.reshape((matrix_size, 1))))
    mask_out = Variable(torch.FloatTensor(mask_out.reshape((matrix_size, 1))))
    mask_loop = Variable(torch.FloatTensor(mask_loop))
    
    if USE_CUDA:
        adj_arc_in = adj_arc_in.cuda()
        adj_arc_out = adj_arc_out.cuda()
        adj_lab_in = adj_lab_in.cuda()
        adj_lab_out = adj_lab_out.cuda()
        
        mask_in = mask_in.cuda()
        mask_out = mask_out.cuda()
        mask_loop = mask_loop.cuda()
        
    if pos_instance:
        pos_instance += batch_size
        
    return pos_instance, input_var, input_lengths, target_var, target_lengths,\
            adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop,\
            id_pairs

def variable_from_sentence(lang, sentence, USE_CUDA=False):
    indexes = indexes_from_sentence(lang, sentence)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if USE_CUDA: var = var.cuda()
    return var

def variables_from_pair(pair, input_lang, output_lang, USE_CUDA=False):
    input_variable = variable_from_sentence(input_lang, pair[0], USE_CUDA)
    target_variable = variable_from_sentence(output_lang, pair[1], USE_CUDA)
    return (input_variable, target_variable)

###################### LANG'S ############################

class Lang:
    def __init__(self):
        self.trimmed = False
        self.stoi = {}
        self.word2count = {}
        self.itos = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.stoi:
            self.stoi[word] = self.n_words
            self.word2count[word] = 1
            self.itos[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.stoi), len(keep_words) / len(self.stoi)
        ))

        # Reinitialize dictionaries
        self.stoi = {}
        self.word2count = {}
        self.itos = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

        for word in keep_words:
            self.index_word(word)
            
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def normalize_pairs(pairs):
    for pair in pairs:
        pair[0] = normalize_string(pair[0])
        pair[1] = normalize_string(pair[1])

def filter_pairs(pairs, max_length):
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) <= max_length \
            and len(pair[1]) <= max_length:
                filtered_pairs.append(pair)
    return filtered_pairs

def prepare_data(pairs_train, pairs_test, max_length):
    
    normalize_pairs(pairs_train)
    normalize_pairs(pairs_test)
    print("Reading pairs %d" % len(pairs_train))
    
    pairs_train = filter_pairs(pairs_train, max_length)
    pairs_test = filter_pairs(pairs_test, max_length)
    print("Filtered to %d pairs" % len(pairs_train))
    
    sentence =  Lang()
    sense = Lang()
    
    print("Indexing words...")
    for pair in pairs_train:
        sentence.index_words(pair[0])
        sense.index_words(pair[1])
    
    for pair in pairs_test:
        sentence.index_words(pair[0])
        sense.index_words(pair[1])
    
    print('Indexed %d words in input language, %d words in output' % (sentence.n_words, sense.n_words))
    return sentence, sense, pairs_train, pairs_test

############## GENERATE VECTORS ##########################
def remove_char(pairs):
    rem_pairs = []
    for pair in pairs:
        pair[0] = re.sub(r"[?|¿|;|!|\(|\)|'|:|%|=|*|+]", '', pair[0])
        pair[0] = pair[0].replace('/', ' ')
        pair[0] = pair[0].replace('-', ' ')
        
        pair[1] = re.sub(r"[?|¿|;|!|\(|\)|'|:|%|=|*|+]", '', pair[1])
        pair[1] = pair[1].replace('/', ' ')
        pair[1] = pair[1].replace('-', ' ')
        rem_pairs.append(pair)  
        
    return np.array(rem_pairs)
    
def construct_vectors(pairs, vector_name_in='fasttext.en.300d', vector_name_out='fasttext.en.300d', fill_rare_words=True):
    
    normalize_pairs(pairs)
    remove_char(pairs)
    
    lang_in = pd.DataFrame(pairs[:, 0], columns=["lang_in"])
    lang_out = pd.DataFrame(pairs[:, 1], columns=["lang_out"])

    lang_in.to_csv('lang_in.csv', index=False)
    lang_out.to_csv('lang_out.csv', index=False)

    lang_in = data.Field(sequential=True, lower=True, init_token="<sos>", eos_token="<eos>")
    lang_out = data.Field(sequential=True, lower=True, init_token="<sos>", eos_token="<eos>")
    
    mt_lang_in = data.TabularDataset(
        path='lang_in.csv', format='csv',
        fields=[('lang_in', lang_in)])
    mt_lang_out = data.TabularDataset(
        path='lang_out.csv', format='csv',
        fields=[('lang_out', lang_out)])

    lang_in.build_vocab(mt_lang_in)
    lang_out.build_vocab(mt_lang_out)

    lang_in.vocab.load_vectors(vector_name_in)
    lang_out.vocab.load_vectors(vector_name_out)
    
    # for rare words in sense vectors, like activated_3281 or something like that
    if fill_rare_words:
        for word in lang_out.vocab.itos:
            if '_' in word:
                rare_word = word.split('_')[0]
                ix_rare_sent = lang_in.vocab.stoi[rare_word]
                ix_rare_sens = lang_out.vocab.stoi[word]
                lang_out.vocab.vectors[ix_rare_sens] = lang_in.vocab.vectors[ix_rare_sent].clone()
    
    return lang_in, lang_out