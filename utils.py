import numpy as np
import time
import math
import json
import codecs
import matplotlib.pyplot as plt

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def save_json(file_path, pairs):
    pairs_list = pairs.tolist() # nested lists with same data, indices
    json.dump(pairs_list, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
def load_json(file_path):
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    a_new = np.array(b_new)
    
    return a_new

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=1) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def plot_losses(train_loss, val_loss, scale):
    plt.figure(figsize=(10,5))
    plt.plot(train_loss)
    plt.plot([(x + 1) * scale - 1 for x in range(len(val_loss))], val_loss)
    plt.legend(['train loss', 'validation loss'])

def print_tree(tree, idx):
    for t in tree.children:
        print(f'arbol: {tree.idx} N hijos: {tree.num_children}')
        print(f'parent: {idx} child {t.idx}')
        print()
        print_tree(t, t.idx)