'''
Created on May 8, 2018

@author: terry
'''
#imports
import numpy
import random
import sys
from nltk.corpus import wordnet as wn

#self-packages
from default import bench_data
from default import semantic_parser as sp
from default import sim_calc

def build_word_data(words, model, refi_flag):
    words_list = list() 
    for word in words:
        
        if refi_flag:
            temp_data_packlist = build_synset_packages_refi(word, model)  
        else:
            temp_data_packlist = build_synset_packages(word)
        
        
        tmp_wordsdata = bench_data.WordsData(word)#initialize WordsData item
        tmp_wordsdata.wndatapack = temp_data_packlist #for each WORD there is a bunch of synsets associated
        words_list.append(tmp_wordsdata)

    return(words_list)    
#create a list of WordsData based on a iterable list of WORDS - Use WNData


def build_synset_packages(word, *pos):
    synset_wndata_list = [] #list of WNDATA{}
#maybe deal with words that are not in the model
    if not pos:#for all POS
        synsets = sp.synset_all(word)  
    else:#for specific POS
        synsets = sp.synset_pos(word,pos)  
    
    for sys_element in synsets:
        synset_wndata_list.append(bench_data.WNData(sys_element, sys_element.offset(), sys_element.pos(), sys_element.definition()))
    
    return(synset_wndata_list)
#list of SYNSET:OFFSET:POS:GLOSS based on DefiData

def build_synset_packages_refi(word, embed_model, *pos):
    synset_refidata_list = []

    if not pos:#for all POS
        synsets = sp.synset_all(word)
    else:#for specific POS
        synsets = sp.synset_pos(word,pos)
    
    for synset in synsets:#create list of Synsets, offset, POS
        key = sp.key_parser(word, synset)
        wpack = bench_data.WNData(synset, synset.offset(), synset.pos(), synset.definition())
        vec = sp.retrieve_synsetvec(key, embed_model)
        wpack.vector = vec
        synset_refidata_list.append(wpack)
   
    return(synset_refidata_list)
#list of SYNSET:OFFSET:POS:GLOSS based on DefiData

def gloss_average_np(words_data, trained_w2v_model, refi_flag):       
    temp_dict = {}#temporary dictionary to avoid repetitive gloss-vector average for each document(words-data)
    for w in words_data: #word-token from text
        for wn_data in w.wndatapack: #list of synsets with their offset/pos/glosses
            if refi_flag: #in case synse2vec model used gloss_vec_acg is 'ignored' with [0.0]
                wn_data.gloss_avg_vec = numpy.full(trained_w2v_model.vector_size, 0.0, dtype=float)
            else: 
                if wn_data.sys_id in temp_dict: #check if the synset for this document already has an average-calculated vector
                    wn_data.gloss_avg_vec = temp_dict.get(wn_data.sys_id)
                else: #if this is the first time the synset is evaluated calculate its average-gloss-vector        
                    gloss_tokens = sp.tokenize_text(wn_data.gloss) #tokenize gloss
                    vecs = []
                    for gloss_token in gloss_tokens:
                        try:
                            vec = trained_w2v_model.word_vec(gloss_token) #return the vector for the token in the gloss
                            vecs.append(vec) #make a list of all token-vector from the word embedding
                        except KeyError:
                            pass
                        
                    #in case all the tokens in the gloss vector do not exist in model we 
                    #mark that gloss average as [0.0] (same dim as model)
                    if vecs:
                        wn_data.gloss_avg_vec = numpy.average(vecs, axis=0)
                    else:
                        wn_data.gloss_avg_vec = numpy.full(trained_w2v_model.vector_size, 0.0, dtype=float)  
    
                    temp_dict[wn_data.sys_id] = wn_data.gloss_avg_vec #add its synset:average-gloss-vector for later lookup
    return(words_data)           
#calculates the average dim-value of the words in the gloss of every word that exists in a word2vec model

def make_bsd(words_data,refi_flag):
    last_index = (len(words_data)-1)
   
    if(len(words_data)==1):#if single-word-document pick MCS to represent it
        only_sys = wn.synsets(words_data[last_index].word)  # @UndefinedVariable
        sys = bench_data.PrimeData(only_sys[0], only_sys[0].offset(), only_sys[0].pos())
        words_data[last_index].prime_sys = sys
    else:
        for index, wd in enumerate(words_data):
            current = wd.wndatapack
            alfa = 0.0
            beta = 0.0
            if (index > 0) and (index < last_index ) : #middle words
                former = words_data[index-1].wndatapack
                latter = words_data[index+1].wndatapack
                if not refi_flag:
                    alfa, sys_a = defidata_gloss_avg_handler(current, former)
                    beta, sys_b = defidata_gloss_avg_handler(current, latter)
                else:
                    alfa, sys_a = refidata_vector_handler(current, former)
                    beta, sys_b = refidata_vector_handler(current, latter)     
            elif index == 0: #first word
                #former = None
                latter = words_data[index+1].wndatapack
                alfa, sys_a = defidata_gloss_avg_handler(current, latter) if not refi_flag else refidata_vector_handler(current, latter) 
            else:#last word
                #latter = None
                former = words_data[index-1].wndatapack
                alfa, sys_a = defidata_gloss_avg_handler(current, former) if not refi_flag else refidata_vector_handler(current, former)
                
            #pick the highest cosine-prime_obj   
            if alfa > beta:
                sysd = sys_a
            elif beta > alfa:
                sysd = sys_b
            else:
                pick_random = [sys_a, sys_b]
                sysd = random.choice(pick_random)
            
            wd.prime_sys = sysd
        
    return(words_data)    
#evaluates which SID represents a word considering its context of +1 and -1

def defidata_gloss_avg_handler(prime, not_prime):
    highest_so_far = sys.float_info.min #value of dist.cost (1 - cosine.similarity) to initialize    
    sys_prime = prime[0].sys_id #just give a Synset to initialize it
    offset_prime = prime[0].offset
    pos_prime = prime[0].pos
    
    #keep the synsetdata with the highest cosine
    for current in prime:
        for evaluated in not_prime:
            tmp_highest = sim_calc.cosine_similarity(current.gloss_avg_vec, evaluated.gloss_avg_vec) #using gloss-average cosine
            if tmp_highest >= highest_so_far:
                highest_so_far = tmp_highest
                sys_prime = current.sys_id
                offset_prime = current.offset
                pos_prime = current.pos
    
    prime_obj = bench_data.PrimeData(sys_prime, offset_prime, pos_prime)
    return (highest_so_far, prime_obj)
#Given two DefiData it returns the 'Synset' for the 'current' with the highest cosine dist value

def refidata_vector_handler(prime, not_prime):
    highest_so_far = sys.float_info.min #value of dist.cost (1 - cosine.similarity) to initialize    
    sys_prime = prime[0].sys_id #just give a Synset to initialize it
    offset_prime = prime[0].offset
    pos_prime = prime[0].pos
    
    #keep the synsetdata with the highest cosine
    for current in prime:
        for evaluated in not_prime:
            tmp_highest = sim_calc.cosine_similarity(current.vector, evaluated.vector) #using gloss-average cosine
            if tmp_highest >= highest_so_far:
                highest_so_far = tmp_highest
                sys_prime = current.sys_id
                offset_prime = current.offset
                pos_prime = current.pos
    
    prime_obj = bench_data.PrimeData(sys_prime, offset_prime, pos_prime)
    return (highest_so_far, prime_obj)
#Given two RefiData it returns the 'Synset' for the 'current' with the highest cosine dist value
