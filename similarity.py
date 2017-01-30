
import numpy as np
import gzip
from collections import defaultdict, Counter
import math
import sys

threshold = 100 #appearance threshold to assign a vector to a word.
contex_thershold = 5 #ignore rare words as context to reduce memory use
co_occure_thres = 2


def make_word_vectors(context_function):
    with gzip.open(data_file, 'r') as fin:
    #with open(sample,'r') as fin:
        paragraph = ''
        for line in fin:
            l = line.rstrip()
            if l: #not empty lines
                paragraph += line
            else:
                add_context(paragraph,context_function)
                paragraph = ''

    global N
    N = 0  # number of all context windows (*,*) in corpus
    for key in features_dict:
        N += sum(features_dict[key].values())


junk = ['det'  , 'PRP$','PRP', 'CC','p']
class word:
    def __init__(self,string):
        self.vals = string.split()
        self.lemma = self.vals[2]
        self.head = int(self.vals[6]) - 1
        self.type = self.vals[7]+'_'
        self.junk = False
        for j in junk:
            if j in self.vals:
                self.junk = True
                break



def make_paragraph(para, trim_function_words = True):
    words =  [word(s) for s in para.splitlines()]
    if trim_function_words:
        return [w for w in words if not w.junk]
    return words

""" context makers """
def all_sentence_context(para):
    return window_context(para, 0)


def window_context(para,window_size=2):
    para = make_paragraph(para)

    #look at the whole sentence.
    if window_size == 0:
        window_size = len(para)

    result = []
    for i in range(len(para)):
        k = min(len(para) , window_size+ i +1) #dont go out of array index boundary.
        w_i = para[i].lemma
        for j in range(i+1,k):
            w_j = para[j].lemma
            result.append((w_i,w_j))
            result.append((w_j, w_i))
    return sorted(result)


def dependency_context(para):
    result = []
    para = make_paragraph(para,False)
    for i in range(len(para)):
        w_i = para[i].lemma
        head_index = para[i].head
        if head_index < 0: continue
        label = para[i].type
        edge1 = (w_i , 'back_'+label+para[head_index].lemma )
        edge2 = (para[head_index].lemma , 'forward_'+label+w_i)
        result.append(edge1)
        result.append(edge2)

        if para[head_index].type == 'adpmod':
            prp = para[head_index]
            child = para[i]
            parent = para[prp.head]
            edge1 = (parent.lemma , prp.lemma+'_' + child.lemma)
            edge2 = (child.lemma , 'forward_'+prp.lemma+'_'+parent.lemma)
            result.append(edge1)
            result.append(edge2)

    return  sorted(result)






def count_words(paragraph):
    para = make_paragraph(paragraph)
    for word in para:
        counts[word.lemma] += 1



def add_context(para,context_maker):
    for word,context in context_maker(para):
        if counts[word] >= threshold and counts[context.rsplit('_')[-1]] >= contex_thershold:   # and counts[context] >= contex_thershold
            context_counts_for_word = features_dict[word]
            context_counts_for_word[context] += 1

        counts_in_window[word] += 1



def top_n_from_dict(dic,n=21):
    as_list = zip(dic.itervalues(), [key for key in dic])
    as_list = sorted(as_list , reverse=True)
    return as_list[0:n]



"""gets the a dict where each entry is a feature vector of a word(also as a dict) where each feature
has the number of co-occureence of the feature(context) with the word.
for each such feature vector(dict), replaces the value to be the PMI of the context with the word."""
def calc_PMI(features_dict, symetric = True):

    for y in features_dict:
        for x in features_dict[y]:
            p_x_given_y = float(features_dict[y][x]) / counts_in_window[y]
            t = x.rsplit('_')[-1]
            p_x = float(counts_in_window[t]) / N
            pmi = p_x_given_y / p_x
            if symetric:

                pmi = pmi / 2
            features_dict[y][x] = math.log(pmi, 2)


def normalize_features(features_dict):
    def normalize(vec):
        result = sum([val**2 for val in vec.itervalues()])
        norm = 1.0/math.sqrt(result)
        for key in vec:
            vec[key] = vec[key] * norm

    for word in features_dict:
        normalize(features_dict[word])


def spare_matrix_mult(matrix,vector):
    d = defaultdict(float)
    for key1 in vector:
        val = vector[key1]
        row = matrix[key1]
        for key2 in row:
            d[key2] += row[key2] *val

    return d


def matrix_mult(matrix,vector):
    d = defaultdict(int)
    for key in matrix:
        row = matrix[key]
        for bla in row:
            if bla in vector:
                d[key] += row[bla]*vector[bla]
    return d


""" remove all keys with value less than thershold"""
def clean_features(features_dict,threshold):
    for key in features_dict.keys():
        vec = features_dict[key]
        for key2 in vec.keys():
            if vec[key2] < threshold:
                del vec[key2]

        if not vec.keys():
            del features_dict[key]


counts = defaultdict(int)


context_type = 1 #0/1/2 for sentence/window/depend
if len(sys.argv) > 1:
    context_type = int(sys.argv[1])
data_file = 'wikipedia.sample.trees.lemmatized.gz'

#count words first
with gzip.open(data_file, 'r') as fin:
#with open(sample,'r') as fin:
    paragraph = ''
    for line in fin:
        l = line.rstrip()
        if l: #not empty lines
            paragraph += line
        else:
            count_words(paragraph)
            paragraph = ''




tog = context_type

funcs = [all_sentence_context,  window_context , dependency_context]
#contex_functions = [all_sentence_context , window_context]
contex_functions = [funcs[tog]]
words = 'car bus hospital hotel gun bomb horse fox table bowl guitar piano '.split()
to_print = defaultdict(list)

for function in contex_functions:
    counts_in_window = Counter()  # count for each word how many windows it shows in for calculating PMI later
    features_dict = defaultdict(Counter)
    N = 0  # number of all context windows (*,*) in corpus

    make_word_vectors(function)
    clean_features(features_dict,co_occure_thres)

    calc_PMI(features_dict)
    #clean_features(features_dict, threshold=-0.1)

    normalize_features(features_dict)

    #print matrix dimensions
    d = set()
    d1 = len(features_dict)
    d2 = []
    for key in features_dict:
        for key2 in features_dict[key]:
            d2.append(features_dict[key][key2])
    print 'Matrix Dim ' + str(len(set(d2))) + ' X ' + str(d1)

    for w in words:
        vector = features_dict[w]
        #d = spare_matrix_mult(features_dict , vector)
        d = matrix_mult(features_dict,vector)
        to_print[w].append(top_n_from_dict(d))






""" printing results"""
names = ['sentence context', 'window=2 context','dependency']
for word in words:
    a , b = to_print[word][0] , to_print[word][1]
    a = to_print[word][0]
    print '------------------- ' + word + '-------------------------'
    print names[tog]
    for i in range(len(a)):
        print str(a[i][1]) + '      ' + str(b[i][1])
