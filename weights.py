import numpy as np
import collections

def log_weights(dct, V):
    '''
    dct: Gensim dictionary
    V: vocabulary size (int)
    '''
    
    ordered_dct = collections.OrderedDict(sorted(dct.cfs.items()))
    values = list(ordered_dct.values())
    #weights = [-np.log(x / dct.num_pos) for x in values]
    weights = [x / dct.num_pos for x in values]
    weights = [-np.log(w) if w > 1/V else -np.log(1/V) for w in weights ]
    weights = [(w - np.min(weights)) / (np.max(weights) - np.min(weights)) for w in weights]
    return weights

def pmi_weights(sections_cleaned):
    corpus = sections_cleaned['corpus']
    dct = sections_cleaned['dct']
    
    sections_lens = list(map(lambda x: sum(map(lambda y: y[1],x)),corpus))
    
    ordered_dct = collections.OrderedDict(sorted(dct.cfs.items()))
    values = list(ordered_dct.values())
    weights = {}
    w_max = float('-inf')
    w_min = float('+inf')
    
    for s_idx, s in enumerate(corpus):
        section_length = sections_lens[s_idx]
        
        for w, wc in s:
            weight_section = wc / section_length
    
            weight = -np.log( weight_section / (values[w] / dct.num_pos))
            weights[(s_idx,w)]  = weight
            w_max = max(w_max, weight)
            w_min = min(w_min, weight)
    
    print('normalizing')
    for key,val in weights.items():
        weights[key] = (val - w_min) / (w_max - w_min)
    
    return weights, w_max, w_min

def highres_weights(docs_cleaned, sections_cleaned):
    corpus = sections_cleaned['corpus']
    dct = sections_cleaned['dct']
    paper_ids = sections_cleaned['ids']
    
    corpus_doc = docs_cleaned['corpus']
    paper_ids_doc = docs_cleaned['ids']
    dct_doc = docs_cleaned['dct']
    
    sections_lens = list(map(lambda x: sum(map(lambda y: y[1],x)),corpus))
    document_lens = list(map(lambda x: sum(map(lambda y: y[1],x)),corpus_doc))
    corpus_doc_dictlist = list(map(dict, corpus_doc))
    weights = {} 
    w_max = float('-inf')
    w_min = float('+inf')
    for s_idx,s in enumerate(corpus):  
        section_length = sections_lens[s_idx]
        paper_id = paper_ids[s_idx]
        ind = paper_ids_doc.index(paper_id)
        document_length = document_lens[ind]
        for w, wc in s:
            # Calculate section-level weight
            weight_section = wc / section_length
            
            # Calculate document-level weight
            w_doc = dct_doc.token2id[dct[w]] #what happens when section word isnt in doc dct?    and
            try:
                count_doc = corpus_doc_dictlist[ind][w_doc]
            except:
                count_doc = 0
            weight_document = count_doc / document_length
            
            # Combined weight
            weight_combined = -np.log(weight_section/weight_document)
            w_max = max(w_max, weight_combined)
            w_min = min(w_min, weight_combined)
            weights[(s_idx,w)] = weight_combined

    print('normalizing')
    for key,val in weights.items():
        weights[key] = (val - w_min) / (w_max - w_min)
    
    return weights

# Update gensim corpus with weighted values
def update_corpus(corpus, weights, weight_type=''):
    for d_idx,d in enumerate(corpus):

        for w_idx, c in enumerate(d):
            #print(d_idx, w_idx)
            #print(c)
            #print(d_idx,w_idx)
            if weight_type=='log':
                corpus[d_idx][w_idx] = (c[0], float(c[1])*weights[c[0]])
            else:    
                corpus[d_idx][w_idx] = (c[0], float(c[1])*weights[(d_idx,c[0])])
    return corpus