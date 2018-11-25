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

import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
from tree import Tree
import torchtext 
from torchtext import data
from torchtext import datasets

# label of dependencies https://nlp.stanford.edu/pubs/USD_LREC14_paper_camera_ready.pdf

DEP_LABELS = ['ROOT', 'ACL','ACVLCL', 'ADVMOD', 'AMOD', 'APPOS', 'AUX', 'CASE', 'CC', 'CCOMP',
               'CLF', 'COMPOUND', 'CONJ', 'COP', 'CSUBJ', 'DEP', 'DET',
               'DISCOURSE', 'DISLOCATED', 'EXPL', 'FIXED', 'FLAT', 'GOESWITH',
               'IOBJ', 'LIST', 'MARK', 'NMOD', 'NSUBJ', 'NUMMOD',
               'OBJ', 'OBL', 'ORPHAN', 'PARATAXIS', 'PUNXT', 'REPARANDUM', 'VOCATIVE',
               'XCOMP']

_DEP_LABELS_DICT = {label:ix for ix, label in enumerate(DEP_LABELS)}

def find_type(type_dep):
    if type_dep=='NSUBJ' or type_dep=='OBJ' or type_dep=='IOBJ' or type_dep=='CSUBJ' or type_dep=='CCOMP' or type_dep == 'XCOMP':
        return 0
    elif type_dep=='OBL' or type_dep=='VOCATIVE' or type_dep=='DISLOCATED' or type_dep=='ADVCL' or type_dep=='ADVMOD' or type_dep=='DISCOURSE' or type_dep=='AUX' or type_dep=='COP' or type_dep=='MARK':
        return 1
    elif type_dep=='NMOD' or type_dep=='APPOS' or type_dep=='NUMMOD' or type_dep=='ACL' or type_dep=='AMOD' or type_dep=='DET' or type_dep=='CLF' or type_dep=='CASE':
        return 2
    else:
        return 3

def get_adj(deps, batch_size, seq_len, max_degree):

    adj_arc_in = np.zeros((batch_size * seq_len, 2), dtype='int32')
    adj_lab_in = np.zeros((batch_size * seq_len, 1), dtype='int32')
    adj_arc_out = np.zeros((batch_size * seq_len * max_degree, 2), dtype='int32')
    adj_lab_out = np.zeros((batch_size * seq_len * max_degree, 1), dtype='int32')


    mask_in = np.zeros((batch_size * seq_len), dtype='float32')
    mask_out = np.zeros((batch_size * seq_len * max_degree), dtype='float32')

    mask_loop = np.ones((batch_size * seq_len, 1), dtype='float32')

    tmp_in = {}
    tmp_out = {}

    for d, de in enumerate(deps):
        for a, arc in enumerate(de):
            if arc[0] != 'ROOT' and arc[0].upper() in DEP_LABELS:         
                arc_1 = int(arc[2])-1
                arc_2 = int(arc[1])-1
                
                if a in tmp_in:
                    tmp_in[a] += 1
                else:
                    tmp_in[a] = 0

                if arc_2 in tmp_out:
                    tmp_out[arc_2] += 1
                else:
                    tmp_out[arc_2] = 0

                idx_in = (d * seq_len) + a + tmp_in[a]
                idx_out = (d * seq_len * max_degree) + arc_2 * max_degree + tmp_out[arc_2]

                adj_arc_in[idx_in] = np.array([d, arc_2])  # incoming arcs
                adj_lab_in[idx_in] = np.array([find_type([arc[0].upper()])])  # incoming arcs

                mask_in[idx_in] = 1.

                if tmp_out[arc_2] < max_degree:
                    adj_arc_out[idx_out] = np.array([d, arc_1])  # outgoing arcs
                    adj_lab_out[idx_out] = np.array([find_type([arc[0].upper()])])  # outgoing arcs
                    mask_out[idx_out] = 1.

        tmp_in = {}
        tmp_out = {}

    adj_arc_in = Variable(torch.LongTensor(np.transpose(adj_arc_in)))
    adj_arc_out = Variable(torch.LongTensor(np.transpose(adj_arc_out)))

    adj_lab_in = Variable(torch.LongTensor(np.transpose(adj_lab_in)))
    adj_lab_out = Variable(torch.LongTensor(np.transpose(adj_lab_out)))

    mask_in = Variable(torch.FloatTensor(mask_in.reshape((batch_size * seq_len, 1))))
    mask_out = Variable(torch.FloatTensor(mask_out.reshape((batch_size * seq_len, max_degree))))
    mask_loop = Variable(torch.FloatTensor(mask_loop))

    return adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop

def pad_seq(lang, seq, max_length):
    seq += [lang.vocab.stoi['<pad>'] for i in range(max_length - len(seq))]
    return seq

def generate_batches(input_lang, output_lang, batch_size, pairs, arr_dep=[], USE_CUDA=False):
    input_batches = []
    input_trees = []
    target_batches = []
    
    for pos in range(0, len(pairs), batch_size):
        cant = min(batch_size, len(pairs) - pos)
        
        bz = batch_size if (batch_size + pos) < len(pairs) else len(pairs) - pos 
        input_seqs = []
        target_seqs = []
        trees = []
        for i in range(bz):
            ix_pair = pos + i
            input_seqs.append(indexes_from_sentence(input_lang, pairs[ix_pair][0]))
            target_seqs.append(indexes_from_sentence(output_lang, pairs[ix_pair][1]))
            if len(arr_dep):
                trees.append(arr_dep[ix_pair])

        #seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
        #input_seqs, target_seqs = zip(*seq_pairs)

        input_lengths = [len(s) for s in input_seqs]
        input_padded = [pad_seq(input_lang, s, max(input_lengths)) for s in input_seqs]
        target_lengths = [len(s) for s in target_seqs]
        target_padded = [pad_seq(output_lang, s, max(target_lengths)) for s in target_seqs]

        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

        if USE_CUDA:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        
        if len(arr_dep):
            input_trees.append(trees)
        input_batches.append(input_var)
        target_batches.append(target_var)

    if len(arr_dep):
        return input_batches, input_trees, target_batches
    else:
        return input_batches, target_batches

def indexes_from_sentence(lang, sentence):
    return [lang.vocab.stoi[word] for word in sentence.split(' ')] + [lang.vocab.stoi['<eos>']]

def variable_from_sentence(lang, sentence, USE_CUDA=False):
    indexes = indexes_from_sentence(lang, sentence)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if USE_CUDA: var = var.cuda()
    return var

def get_trees(filename):
    trees = [get_tree(array) for array in (np.load(filename))]
    return np.array(trees)

def get_tree(array):
    parents = array
    trees = dict()
    root = None
    for i in range(0, len(parents)):
        if i % 2 == 0:
            idx = i
            parent = parents[idx]
            if parent not in trees.keys():
                tree = Tree()
                trees[parent] = tree
                tree.idx = parent
                if parent == 0:
                    root = tree
        else:
            idx = i
            parent = parents[idx - 1]
            child = parents[idx]
            if child not in trees.keys():
                tree = Tree()
                trees[child] = tree
                tree.idx = child
            else:
                tree = trees[child]
            trees[parent].add_child(tree)
    return root

def get_matrixes(filename):
    trees = [get_matrix(array) for array in (np.load(filename))]
    return np.array(trees)

def get_matrix(array):
    matrix = np.zeros((int(len(array) / 2) + 1, int(len(array) / 2) + 1))
    for i in range(0, len(array), 2):
        if array[i] != 0 or array[i+1] != 0:
            val_iz = array[i] - 1
            val_der =  array[i+1] - 1
            matrix[val_iz][val_der] = 1
            matrix[val_der][val_iz] = 1
            matrix[val_der][val_der] = 1
            matrix[val_iz][val_iz] = 1
        
    return matrix

def construct_vector(pair, name_lang, construct_vector=True, vector_name='fasttext.en.300d', dir='corpus'):
    lang = pd.DataFrame(pair, columns=[name_lang])

    lang.to_csv(f'{dir}/' + name_lang + '.csv', index=False)

    lang = data.Field(sequential=True, lower=True, init_token='<sos>', eos_token='<eos>')

    mt_lang = data.TabularDataset(
        path=f'{dir}/' + name_lang + '.csv', format='csv',
        fields=[(name_lang, lang)])

    lang.build_vocab(mt_lang)

    if construct_vector:
        lang.vocab.load_vectors(vector_name)
    
    return lang
            
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(pair):
    pair = unicode_to_ascii(pair.lower().strip())
    #pair = re.sub(r"([.,;!?'‘’])", r' \1', pair) # separate .!? from words
    
    return ' '.join(pair.split())

def normalize_pairs(pairs):
    for pair in pairs:
        pair[0] = normalize_string(pair[0])
        pair[1] = normalize_string(pair[1])

def filter_pairs_lang(pairs, min_length, max_length):
    filtered_pairs = []
    filtered_indexes = []
    nlp = StanfordCoreNLP(r'data/lib/stanford-corenlp')

    for ix, pair in enumerate(pairs):
        # Removing '' and "" in pairs, this is for easy processing 
        if len(pair[0].split()) >= min_length and len(pair[0].split()) <= max_length \
            and len(pair[1].split()) >= min_length and len(pair[1].split()) <= max_length:
                filtered_pairs.append(pair)
                filtered_indexes.append(ix)
    return filtered_pairs, filtered_indexes

def read_langs(lang1, lang2, reverse=False, dir='corpus'):
    print("Reading lines...")

    # Read the file and split into lines
    filename = f'{dir}/in.txt'
    lines_a = open(filename, encoding='utf8').read().strip().split('\n')
    
    filename = f'{dir}/out.txt'
    lines_b = open(filename, encoding='utf8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l] for l in zip(lines_a, lines_b)]
  
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        
    return pairs

def prepare_data(lang1_name, lang2_name, reverse=False, min_length=0, max_length=50, dir='corpus', return_trees=False, output_tree='matrix'):
    pairs = read_langs(lang1_name, lang2_name, reverse=reverse, dir=dir)
    print("Read %d sentence pairs" % len(pairs))
    
    pairs, indexes = filter_pairs_lang(pairs, min_length, max_length)
    print("Filtered to %d pairs" % len(pairs))
    
    print("Creating vocab...")
    pairs = np.array(pairs)
    indexes = np.array(indexes)
    
    vector_1 = construct_vector(pairs[:, 0], lang1_name, dir=dir)
    vector_2 = construct_vector(pairs[:, 1], lang2_name, dir=dir)
    
    if return_trees:
        if output_tree == 'tree':
            print('Creating trees...')
            input_syntax = get_trees(os.path.join(dir, 'in.parents.npy'))[indexes]
        elif output_tree == 'matrix':
            print('Creating matrixes...')
            input_syntax = get_matrixes(os.path.join(dir, 'in.parents.npy'))[indexes]

    print('Indexed %d words in input language, %d words in output' % (len(vector_1.vocab.itos), len(vector_2.vocab.itos)))
    if return_trees:
        return vector_1, vector_2, input_syntax, output_syntax, pairs
    else:
        return vector_1, vector_2, pairs