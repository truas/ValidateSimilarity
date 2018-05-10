'''
Created on Apr 27, 2018

@author: terry
'''
#general libs
import re
import numpy
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.corpus import wordnet as wn

#my packages
from default import process_synsets as ps
from default import bench_data
from default import wn_synsets_parser as wsp


def key_parser(word, synset):
    key1 = word.lower() +'#'+str(synset.offset())+'#'+synset.pos()
    return(key1)
#parse word/synsets into key format for the model

def validate_synsets_model(word, synsets, trained_model):      
    valid_synsets = []
    for synset in synsets:
        key = key_parser(word, synset)
        try:
            v1 = trained_model.word_vec(key)
            valid_synsets.append(v1) #put all vector words in the sentence together and average
        except KeyError:
            #print(key, "@ is not in the model")
            pass #key not in the model
    return (valid_synsets) #if the key is not on the model, it will return an empty list of valid synets
 #creates a list of synsetkeys that exist in the trained model
 
    
def sentence_adapter(tokens):
    semantic_blocks = []
    for token in tokens:
        tmp_semantic = bench_data.Context_Data(token.word1, token.word2, token.sent1, token.sent2)
        tmp_semantic.ste1 = tokenize_text(token.sent1)
        tmp_semantic.ste2 = tokenize_text(token.sent2)
        semantic_blocks.append(tmp_semantic)
        
    return(semantic_blocks)
#adapt a object-data with 'raw' sentence1 and sentence2 into their 'cleaned-tokenized' versions

def tokenize_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')   
    raw = str(text.lower())
    tokens = tokenizer.tokenize(raw)
    #remove stopwords
    stopped_tokens = [i for i in tokens[:] if not i in en_stop] #[1:] get rid of the first element
    # remove numbers
    nonumber_text = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
    nonumber_text = [i for i in nonumber_text if len(i) > 1]
    nonumber_text = ' '.join(nonumber_text).split()
    
    return(nonumber_text)
#cleans gloss words from numbers and stopwords

#===============================================================================
# WordNet Stuff
#===============================================================================
def wn_sentence_handler(tokens_wn, trained_model):
    wn_token_objects = []
    
    for token_wn in tokens_wn:

        clean_context_a = validate_context_wn(token_wn.ste1)
        clean_context_b = validate_context_wn(token_wn.ste2)

        if not clean_context_a:
            context_a_representation = None
        else:
            pcontex_a = wsp.build_word_data(clean_context_a)
            context_elems_a = wsp.gloss_average_np(pcontex_a, trained_model)
            context_bsds_a = wsp.make_bsd(context_elems_a)
            context_a_representation = wn_key_parser(context_bsds_a)
        
        if not clean_context_b:
            context_b_representation = None
        else:
            pcontex_b = wsp.build_word_data(clean_context_b)
            context_elems_b = wsp.gloss_average_np(pcontex_b, trained_model)
            context_bsds_b = wsp.make_bsd(context_elems_b)
            context_b_representation = wn_key_parser(context_bsds_b)
        
        wn_token_obj = bench_data.WN_Token_Data(token_wn.word1, token_wn.word2)
        wn_token_obj.context1 = context_a_representation
        wn_token_obj.context2 = context_b_representation
        wn_token_objects.append(wn_token_obj)
        
    return(wn_token_objects)
#transforms sentences in sequences of synset-keys following BSD Algorithm

def validate_context_wn(sentence_tokens):
    valid_tokens = []
    for sentence_token in sentence_tokens: 
        synsets = wn.synsets(sentence_token)  # @UndefinedVariable
        if not synsets:
            continue
        else:
            valid_tokens.append(sentence_token)
            
    return(valid_tokens)    
#take elements that only exist in wordnet  

def validate_context_model(sentence_tokens, trained_model):
    valid_tokens = []
    for sentence_token in sentence_tokens: 
        try:
            synsets = wn.synsets(sentence_token)  # @UndefinedVariable
            valid_tokens.append(sentence_token)
        except KeyError:
            pass
    return(valid_tokens)    
#take elements that only exist in the model   
 
def wn_key_parser(words_data):
    bsd_items = []
    for item_word_data in words_data:
        key = item_word_data.word.lower() +'#'+str(item_word_data.prime_sys.poffset)+'#'+item_word_data.prime_sys.ppos
        bsd_items.append(key)
    #key1 = word.lower() +'#'+str(synset.offset())+'#'+synset.pos()
    return(bsd_items)
#parse word/synsets into key format for the model - for WN-BSD entries
 
def naive_wnkey_parser(word, offset, pos):
    return(word.lower() +'#'+str(offset)+'#'+pos)
#Transforms word,offset,pos into key that is used in synset2vec model
 
def retrieve_synsetvec(key, model):
    try:
        tmp_vec = model.word_vec(key)
    except KeyError:
        tmp_vec = [0.0] #key not in the model we set as 0.0
    return(tmp_vec)
#returns the dimension/values of a key in a word-embedding model

def synset_all(word):
    return wn.synsets(word)  # @UndefinedVariable
#synsets for all POS

def synset_pos(word, cat):
    return wn.synsets(word, cat)  # @UndefinedVariable
#synsets for specific POS       
    