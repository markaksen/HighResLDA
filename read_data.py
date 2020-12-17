import numpy as np
import pandas as pd
import io
import json
import gzip
from tqdm import tqdm
import os
import pickle

def process_metadata(metadata_file,fields=None, get_ids=None,output_dir = './processed/',put_in_op_dir=False):
    '''
    Load metadata from json file (compressed ok) filtered by get_ids or field of paper into pandas DataFrame and save it.

    Parameters
    ----------
    metadata_file : TYPE
        DESCRIPTION.
    fields : TYPE, optional
        DESCRIPTION. The default is None.
    get_ids : TYPE, optional
        DESCRIPTION. The default is None.
    output_dir : TYPE, optional
        DESCRIPTION. The default is './processed/'.
    put_in_op_dir : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    meta_df_file : TYPE
        DESCRIPTION.

    '''

    file_name_without_path_or_ext = metadata_file.split('/')[-1].split('.')[0]
    # Go through metadata files to get relevant paper ids and titles
    ids = []; title = []; field = [];
    # if file is compressed
    if metadata_file[-3:] == '.gz':
        if put_in_op_dir:
            output_file = output_dir + file_name_without_path_or_ext
        else:  
            output_file = metadata_file[:-3]
        gz = gzip.open(metadata_file, 'rb')
        f = io.BufferedReader(gz)
        f_out = open(output_file,'wb')
    else:
        f = open(metadata_file)
        f_out = None

    for line in tqdm(f.readlines()):
        paper = json.loads(line)
        if not fields:
            if not get_ids:
                ids.append(paper['paper_id'])
                title.append(paper['title'])
                field.append(paper['mag_field_of_study'])
                if f_out:
                    f_out.write(line)
            elif paper['paper_id'] in get_ids:
                ids.append(paper['paper_id'])
                title.append(paper['title'])
                field.append(paper['mag_field_of_study'])
                if f_out:
                    f_out.write(line)
        elif paper['mag_field_of_study']:
            field_in = any([x in fields for x in paper['mag_field_of_study']])
            if field_in:
                ids.append(paper['paper_id'])
                title.append(paper['title'])
                field.append(paper['mag_field_of_study'])
                if f_out:
                    f_out.write(line)
    f.close()
    if f_out:
        f_out.close()
    # create and save dataframe in output_dir/meta_df
    meta_df = pd.DataFrame({'ids':ids, 'titles':title, 'field': field})
    meta_df_dir = output_dir + 'meta_df/'
    os.makedirs(meta_df_dir, exist_ok=True)
    meta_df_file = meta_df_dir + file_name_without_path_or_ext + '.pkl'
    with open(meta_df_file, 'wb') as f:
        pickle.dump(meta_df, f)
    return meta_df_file

def process_pdf(meta_df_file, pdf_file, fields=None,get_ids=None, output_dir = './processed/'):
    '''
    Load parsed pdf json file (compressed ok) into pd DF, filter based on  ids in corresponding metadata file.

    'fields' and 'get_ids' could be used in the future to further filter. Not implemented currently.

    Parameters
    ----------
    meta_df_file : TYPE
        DESCRIPTION.
    pdf_file : TYPE
        DESCRIPTION.
    fields : TYPE, optional
        DESCRIPTION. The default is None.
    get_ids : TYPE, optional
        DESCRIPTION. The default is None.
    output_dir : TYPE, optional
        DESCRIPTION. The default is './processed/'.

    Returns
    -------
    text_df_file : TYPE
        DESCRIPTION.

    '''
    
    # use meta data df to check ids
    with open(meta_df_file, 'rb') as f:
        meta_df = pickle.load(f)
    # get the pdfs 
    # lists to make pdf dataframe from 
    papers_ids_text = []; abstract = [] 
    body_text = []  # list of dicts (for each paper) with section, text, cite_spans
    whole_text = [] # list of strings (for each paper) of entire text in the body
    key_words = []  # key words mentioned with the abstract
    citations = [] 
    # if file is compressed
    if pdf_file[-3:] == '.gz':
        output_file = pdf_file[:-3]
        gz = gzip.open(pdf_file, 'rb')
        f = io.BufferedReader(gz)
        f_out = open(output_file,'wb')
    else:
        f = open(pdf_file)
        f_out = None
        
    for line in tqdm(f.readlines()):
        paper = json.loads(line)
        if paper['paper_id'] in meta_df['ids'].values: #ids defined in meta data df based on field
            if f_out: # if untaring
                f_out.write(line) # write that pdf (untarred pdf parse will only have selected papers)
            # get paper-id
            papers_ids_text.append(paper['paper_id'])
            abstract_text = ''
            terms = []
            # get abstract
            if paper['abstract']:
                abstract_text = paper['abstract'][0]['text']
                # get key_words in abstract
                if len(paper['abstract'])>1:   
                    if paper['abstract'][1]['text'][:11].lower() == 'index terms':
                        terms = paper['abstract'][1]['text'][12:].split(',') #remove "Index Terms-" or "INDEX TERMS " from string    
            abstract.append(abstract_text) 
            key_words.append(terms)
            # get sections and text from body
            text = []
            full_text = ''
            if paper['body_text']:
                for entry in paper['body_text']:
                    if entry['section'] and entry['text']:
                        section = {key: entry[key] for key in ['section', 'text', 'cite_spans']}
                        text.append(section)
                        if full_text: # why the if-else?
                            full_text = full_text + '\n' + entry['text']
                        else:
                            full_text = entry['text']
            body_text.append(text)
            whole_text.append(full_text)
            # get citations
            bib_entries = {}
            if paper['bib_entries']:
                bib_entries = paper['bib_entries']
            citations.append(bib_entries)
            
    f.close()
    if f_out:
        f_out.close()
    # create and save dataframe in output_dir/text_df
    text_df = pd.DataFrame({'paper_id': papers_ids_text, 'abstract': abstract,'key_words': key_words,'body_text': body_text, 'whole_text': whole_text,'citations':citations})
    text_df_dir = output_dir + 'text_df/'
    os.makedirs(text_df_dir, exist_ok=True)
    file_name_without_path_or_ext = pdf_file.split('/')[-1].split('.')[0]
    text_df_file = text_df_dir + file_name_without_path_or_ext + '.pkl'
    with open(text_df_file, 'wb') as f:
        pickle.dump(text_df, f)         
    return text_df_file       

def process_batch(metadata_file,pdf_file, fields=None, get_ids=None, output_dir = './processed/'):
    '''
    Process metadata and pdf parses filtered by fields (x)or Ids and save dataframes to output_dir.
    Parameters
    ----------
    metadata_file : TYPE
        DESCRIPTION.
    pdf_file : TYPE
        DESCRIPTION.
    fields : TYPE, optional
        DESCRIPTION. The default is None.
    get_ids : TYPE, optional
        DESCRIPTION. The default is None.
    output_dir : TYPE, optional
        DESCRIPTION. The default is './processed/'.

    Returns
    -------
    meta_df_file : TYPE
        DESCRIPTION.
    text_df_file : TYPE
        DESCRIPTION.

    '''
    meta_df_file = process_metadata(metadata_file,fields,get_ids,output_dir)
    text_df_file = process_pdf(meta_df_file,pdf_file,fields,get_ids,output_dir)
    return meta_df_file,text_df_file

def pappu():
    print("pappu")

def with_keywords(df):
    key_words = df['key_words'][df.key_words.str.len() > 0]
    return df.loc[key_words.index]  # papers with key words


def get_links(paper_refs):
    'Retreive the paper_ids (links) from refrences of the paper'
    refs, links = [], []
    for key,value in paper_refs.items():
        if value['link']:
            links.append(value['link'])
            refs.append(key)
    return links, refs
