#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from datetime import datetime
import sys

start_time = datetime.now()
f_data = ""
e_data = ""

loops = 7
threshold = 0.30
num_sentences = 100000


bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:num_sentences]

#getting unique english and french word
unique_fn = set()
unique_en = set()

for i, (french_sen,english_sen) in enumerate(bitext):
    for word in french_sen:
        unique_fn.add(word)
    for word in english_sen:
        unique_en.add(word)

#saving index to use in numpy
index_map_en = {e:i for i,e in enumerate(unique_en)}
index_map_fn = {f:i for i,f in enumerate(unique_fn)}

# intilialization of translation probability uniformly
a = datetime.now()
translation_probability = np.ones([len(unique_en), len(unique_fn)], dtype=np.float32)
translation_probability = np.divide(translation_probability, len(unique_fn))
#print("Initialization time takes:", (datetime.now()-a).total_seconds())

a = datetime.now()
#EM
for i in range(0,loops):
    count_pair=np.zeros([len(unique_en), len(unique_fn)], dtype=np.float32)
    count_fn = np.zeros([1,len(unique_fn)], dtype=np.float32)
    #expectation
    for i,(f,e) in enumerate(bitext):
        index_f = [index_map_fn[k] for k in f]
        index_e = [index_map_en[k] for k in e]
        temp = translation_probability[np.ix_(index_e,index_f)].copy()
        normalization_term = temp.sum(axis=1)
        adding_term = temp/normalization_term.reshape(-1,1)
        count_pair[np.ix_(index_e, index_f)] = count_pair[np.ix_(index_e, index_f)] + adding_term
        count_fn[:,np.ix_(index_f)] = count_fn[:,np.ix_(index_f)] +adding_term.sum(axis=0)
    
    #maximization
    np.divide(count_pair,count_fn, out=translation_probability)

print("EM time takes:", (datetime.now()-a).total_seconds())
#converting into df
translation_probability = pd.DataFrame(translation_probability, columns=unique_fn, index=unique_en)

#alignment
for (f, e) in bitext:
    for (i, f_i) in enumerate(f):
        best_prob = 0
        best_j = 0
        for (j, e_j) in enumerate(e):
            if translation_probability.loc[e_j,f_i] > best_prob:
                best_prob = translation_probability.loc[e_j,f_i]
                best_j = j
        if best_prob > threshold:
            sys.stdout.write("%i-%i " % (i,best_j))
    sys.stdout.write("\n")

print("total time taken for whole program:", (datetime.now()-start_time).total_seconds())



