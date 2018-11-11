import os
import glob
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
import itertools
import corenlp

os.environ["CORENLP_HOME"] = '/home/krivas/projects/neural-wsd/new_experiments/data/lib/stanford-corenlp'

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def dependency_parse(filepath,  client, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    parentpath = os.path.join(dirpath, filepre + '.parents')
    deps = []
    with open(filepath) as file:
        for line in tqdm(file, total=file.tell()):
            temp = client.dependency_parse(line)
            temp = list(map(lambda x: [int(x[1]), int(x[2])], temp))
            temp = list(itertools.chain(*temp))
            deps.append(temp)
    np.save(parentpath, np.array(deps))

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def split(filepath, dst_dir, client):
    with open(filepath) as datafile, \
            open(os.path.join(dst_dir, 'a.txt'), 'w') as afile, \
            open(os.path.join(dst_dir, 'b.txt'), 'w') as bfile:
        datafile.readline()
        for line in datafile:
            a, b = line.strip().split('\t')

            ann = client.annotate(a)
            s = ' '.join([w.word for w in ann.sentence[0].token])
            afile.write(a + '\n')
                
            ann = client.annotate(b)
            s = ' '.join([w.word for w in ann.sentence[0].token])
            bfile.write(b + '\n')

def parse(dirpath, client, cp=''):
    dependency_parse(os.path.join(dirpath, 'a.txt'), client, cp=cp, tokenize=True)
    dependency_parse(os.path.join(dirpath, 'b.txt'), client, cp=cp, tokenize=True)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing dataset')
    print('=' * 80)

    base_dir = ''
    data_dir = os.path.join(base_dir, 'data')
    all_dir = os.path.join(data_dir, 'translation/all_data')
    lib_dir = os.path.join(base_dir, 'lib')
    train_dir = os.path.join(data_dir, 'translation/train')
    #dev_dir = os.path.join(data_dir, 'translation/dev')
    #test_dir = os.path.join(data_dir, 'translation/test')
    make_dirs([train_dir])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

    # split into separate files
    client = corenlp.CoreNLPClient(annotators="tokenize ssplit".split())
    split(os.path.join(all_dir, 'en-spa.txt'), train_dir, client)
    #split(os.path.join(all_dir, 'SICK_trial.txt'), dev_dir)
    #split(os.path.join(all_dir, 'SICK_test_annotated.txt'), test_dir)

    # parse sentences
    client = StanfordCoreNLP(r'data/lib/stanford-corenlp')
    parse(train_dir, client, cp=classpath)