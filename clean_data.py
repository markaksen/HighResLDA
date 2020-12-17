import pandas as pd
from nltk import pos_tag
from nltk.corpus import wordnet
import gensim
import spacy
import time
import re
import itertools
import gensim.corpora as corpora
import collections
import os
import pickle

def get_word_postag(word):
    #if pos_tag([word])[0][1].startswith('J'):
    #    return wordnet.ADJ
    #if pos_tag([word])[0][1].startswith('V'):
    #    return wordnet.VERB
    if pos_tag([word])[0][1].startswith('N'):
        #return wordnet.NOUN
        return True
    else:
        return False
        #return wordnet.ADJ
        #return wordnet.NOUN

from nltk.tokenize import word_tokenize
# Preprocessing: tokenize words
def tokenize(text):
    return(word_tokenize(text))


def sent_to_words(sentences):
    for sentence in sentences:
        return(gensim.utils.simple_preprocess(str(sentence), min_len=3,deacc=True))  # deacc=True removes punctuations

from nltk.corpus import stopwords
stopwords = stopwords.words('english')

# Preprocessing: remove stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stopwords]) 
    #return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Preprocessing: lemmatizing
nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Preprocessing: remove short text
def find_longer_text(texts,k=200):
    return list(map(lambda x: len(x.split())>k,texts))
    
#     lengths = list(map(lambda x: len(x.split()), texts))
#     return [val >= k for val in lengths]
    #return [idx for idx, val in enumerate(lengths) if val >= k] 

# Preprocessing: alpha num
def keep_alphanum(words):
    #def isalphanum(word):
    #return word.isalnum()
    return filter(lambda word: word.isalnum(), words)
    #return [word for word in words if word.isalnum()]

# Preprocessing: keep nouns
def keep_nouns(words):
    return filter(get_word_postag, words)
    #return [word for word in words if get_word_postag(word) =='n']

# Preprocessing: keep words >= 3 in length
def keep_longer_words(words):
    return list(filter(lambda x: (len(x) >= 3), words))
    #return [word for word in words if len(word) >= 3]

# Preprocessing: lemmatize
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
def lemmatize(words):
    return list(map(lm.lemmatize, words)) 

# Preprocessing: stemming
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 
def stemming(words):
    #return [ps.stem(word) for word in words]
    return map(ps.stem, words)

def remove_digits(words):
    return list(filter(lambda x: x.isalpha(), words))
    #return list(filter(lambda x: x.isalpha(), words))
#     return [word for word in words if word.isalpha()]

def merged(words):
    return ' '.join(word for word in words)   

def clean_pdf(text_df, file_name='', output_dir='',section_lvl = False):
    
    start = time.time()
    

    # if index is not paper_id
    if text_df.index.name is None: 
        print('changing index to paper_id')
        text_df = text_df.set_index('paper_id')

    ids = text_df.index.values.astype(str)
        
    contents = text_df['whole_text'].values.tolist()
    
    # Add abstract to text
    if not section_lvl:
        abstracts = text_df['abstract'].values.tolist()

        contents = [i + j for i, j in zip(contents, abstracts)]
    
    t = time.time()
    print(t-start)
    
    # Remove new line characters
    contents = map(lambda x: re.sub('\s+', ' ', x), contents)
    
    t = time.time()
    print(t-start)
    
    # Preprocessing: lower case text
    contents = map(lambda x: x.lower(),contents)
    
    t = time.time()
    print(t-start)
    
    # Preprocessing: keep alphanumeric
    contents = map(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', x), contents)
    
    t = time.time()
    print(t-start)
    
    # Preprocessing: remove stand along numbers
    contents = map(lambda x: re.sub(" \d+ ", " ", x), contents)

    t = time.time()
    print(t-start)
    
    # Preprocessing: remove stop words
    contents = map(remove_stopwords, contents)
    
    t = time.time()
    print(t-start)
    
    contents = list(contents)

    # Preprocessing: remove short text
    inds = find_longer_text(contents)
    contents = itertools.compress(contents, inds)
    ids = list(itertools.compress(ids, inds))
    
    key_words = text_df.loc[ids]['key_words'].values
    print('Tokenizing')
    
    # Tokenize words + remove punctuation
    tokenized_contents = map(tokenize,contents) # documents in BOW format
#     word_list = [tokenize(article) for article in contents]
    t = time.time()
    print(t-start)
    
    # Remove numbers
    tokenized_contents = map(remove_digits, tokenized_contents)
    
    
    t = time.time()
    print(t-start)
    
    # Keep longer words
#     word_list = [keep_longer_words(words) for words in  word_list]
    tokenized_contents = map(keep_longer_words,  tokenized_contents)
    
    
    t = time.time()
    print(t-start)
    
    print('Lemmatizing')
    
    # Preprocessing: lemmatize
    tokenized_contents = map(lemmatize, tokenized_contents)
    
#     print(list(word_list)[0])
    
    t = time.time()
    print(t-start)
    
    print('Bag of Words Representation')
    tokenized_contents = list(tokenized_contents)

    dct = corpora.Dictionary(tokenized_contents) # make dct before corpus
#     doc2bow = partial(dct.doc2bow,allow_update=True)
    
    print('length of dct before filter_extreme: ', len(dct))
    dct.filter_extremes() # using default params # filter dct before creating corpus
    print('length of dct after filter_extreme: ', len(dct))
    

    # Make corpus after any changes to dct
    corpus = list(map(lambda x: dct.doc2bow(x,allow_update=True), tokenized_contents))    
#     corpus = [dct.doc2bow(doc, allow_update=True) for doc in word_list]

    t = time.time()
    print(t-start)


    
    word_list =  [item for sublist in tokenized_contents for item in sublist]
    counter=collections.Counter(word_list)
    
    if file_name:
        os.makedirs(output_dir, exist_ok=True)
        output_file_name = output_dir + file_name + '_clean.pkl'


        with open(output_file_name, 'wb') as f:  # Python 3: open(..., 'wb') 
            d = {'dct': dct, 'corpus': corpus, 'docs': tokenized_contents, 
                 'counter': counter, 'ids': ids, 'word_list':word_list,
                 'key_words':key_words}
            pickle.dump(d, f)
    return d

def process_sections(textdf, file_name=''):
    def merge_sections(doc):
        sections = doc['body_text']
        #print(len(sections))
        #print(sections[0])
        if type(sections) is list and len(sections)>0:
            section_titles = [x['section'] for x in sections]
            paper_id = doc['paper_id']
            texts = []; titles = [];
            current_text = sections[0]['text']
            current_title = section_titles[0]
            for i in range(1,len(sections)):
                #print(i)
                if section_titles[i] == current_title:
                    current_text = current_text + ' ' + sections[i]['text']
                else: 
                    texts.append(current_text)
                    titles.append(current_title)
                    current_title = section_titles[i]
                    current_text = sections[i]['text']
            sections_df = pd.DataFrame({'whole_text': texts, 'titles':titles, 'paper_id':paper_id})
            return sections_df
        else:
            return []
    total_df = pd.DataFrame(columns = ['whole_text', 'titles', 'paper_id'])
    for index,row in textdf.iterrows():
        #print(index)
        
        total_df = total_df.append(merge_sections(row))
    return total_df

def clean_section(text_df, file_name='', output_dir='',section_lvl = False):
    
    start = time.time()
    # if index is not paper_id
    if text_df.index.name is None: 
        print('changing index to paper_id')
        text_df = text_df.set_index('paper_id')

    ids = text_df.index.values.astype(str)
        
    contents = text_df['whole_text'].values.tolist()
    
    # Add abstract to text
    if not section_lvl:
        abstracts = text_df['abstract'].values.tolist()

        contents = [i + j for i, j in zip(contents, abstracts)]
    
    t = time.time()
    print(t-start)
    
    # Remove new line characters
    contents = map(lambda x: re.sub('\s+', ' ', x), contents)
    
    t = time.time()
    print(t-start)
    
    # Preprocessing: lower case text
    contents = map(lambda x: x.lower(),contents)
    
    t = time.time()
    print(t-start)
    
    # Preprocessing: keep alphanumeric
    contents = map(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', x), contents)
    
    t = time.time()
    print(t-start)
    
    # Preprocessing: remove stand along numbers
    contents = map(lambda x: re.sub(" \d+ ", " ", x), contents)

    t = time.time()
    print(t-start)
    
    # Preprocessing: remove stop words
    contents = map(remove_stopwords, contents)
    
    t = time.time()
    print(t-start)
    
    contents = list(contents)

    # Preprocessing: remove short text
    inds = find_longer_text(contents)
    contents = itertools.compress(contents, inds)
    ids = list(itertools.compress(ids, inds))
    
    #key_words = text_df.loc[ids]['key_words'].values
    print('Tokenizing')
    
    # Tokenize words + remove punctuation
    tokenized_contents = map(tokenize,contents) # documents in BOW format
#     word_list = [tokenize(article) for article in contents]
    t = time.time()
    print(t-start)
    
    # Remove numbers
    tokenized_contents = map(remove_digits, tokenized_contents)
    
    
    t = time.time()
    print(t-start)
    
    # Keep longer words
#     word_list = [keep_longer_words(words) for words in  word_list]
    tokenized_contents = map(keep_longer_words,  tokenized_contents)
    
    
    t = time.time()
    print(t-start)
    
    print('Lemmatizing')
    
    # Preprocessing: lemmatize
    tokenized_contents = map(lemmatize, tokenized_contents)
    
#     print(list(word_list)[0])
    
    t = time.time()
    print(t-start)
    
    print('Bag of Words Representation')
    tokenized_contents = list(tokenized_contents)

    dct = corpora.Dictionary(tokenized_contents) # make dct before corpus
#     doc2bow = partial(dct.doc2bow,allow_update=True)
    
    print('length of dct before filter_extreme: ', len(dct))
    dct.filter_extremes() # using default params # filter dct before creating corpus
    print('length of dct after filter_extreme: ', len(dct))
    

    # Make corpus after any changes to dct
    corpus = list(map(lambda x: dct.doc2bow(x,allow_update=True), tokenized_contents))    
#     corpus = [dct.doc2bow(doc, allow_update=True) for doc in word_list]

    t = time.time()
    print(t-start)
    
    word_list =  [item for sublist in tokenized_contents for item in sublist]
    counter=collections.Counter(word_list)
    
    if file_name:
        os.makedirs(output_dir, exist_ok=True)
        output_file_name = output_dir + file_name + '_clean.pkl'


        with open(output_file_name, 'wb') as f:  # Python 3: open(..., 'wb') 
            d = {'dct': dct, 'corpus': corpus, 'docs': tokenized_contents, 
                 'counter': counter, 'ids': ids, 'word_list':word_list}
            pickle.dump(d, f)
    return d