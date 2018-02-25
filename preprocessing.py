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
    
    c = join_words(c, word_dict)
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
                print('---oracione sin parentesis')
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

    nlp = StanfordCoreNLP(path_model + "stanford-corenlp-full-2018-01-31/")
    
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
        data = re.sub(r'he\'s', 'he is', data)
        data = re.sub(r'u \'d', 'uld', data)
        data = re.sub(r'&', '', data)
        if(is_train):
            pairs.extend(process_instance(ix_ins, data, ner, parser, word_dict, nlp, is_train, None, prune_sentence, verbose))
        else:
            pairs.extend(process_instance(ix_ins, data, ner, parser, word_dict, nlp, is_train, senses_all[ix_ins], prune_sentence, verbose))
        
    
    return np.array(pairs)