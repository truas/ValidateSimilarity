'''
Created on Apr 27, 2018

@author: terry
'''
#general libs
import re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.corpus import wordnet as wn

#my packages
from default import process_synsets as ps
from default import bench_data


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
        
    