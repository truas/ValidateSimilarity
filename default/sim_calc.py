import numpy
from scipy import spatial
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from nltk.corpus import wordnet as wn
from default import bench_data
from default import semantic_parser as sp


#Global - Definitions
PRECISION_COS = 7
          #0        #1    #2        #3
RANGE = ([-1,1], [0,10], [0,4], [0,50])#0 - COSINE; 1 - WS353/SIMLEX/STANFORD; 2 - RG65; 3 - MEN

#===============================================================================
# NO-CONTEXT SIMILARITY
#===============================================================================

def nocontext_sim(tokens, trained_model, type_range, metric='avgsim'):
    moretokens =[]
    
    for token in tokens:
        word_a = token.word1
        word_b = token.word2
        synsets_a = sp.synset_all(word_a)
        synsets_b = sp.synset_all(word_b)
        
        #clean synsets that only exist in the model
        vec_syna = sp.validate_synsets_model(word_a, synsets_a, trained_model)
        vec_synb = sp.validate_synsets_model(word_b, synsets_b, trained_model)
        
        if metric == 'maxsim':
            sim_value = maxSim(vec_syna, vec_synb, type_range)
        elif metric == 'avgsim':
            sim_value = avgSim(vec_syna, vec_synb, type_range)
        elif metric == 'globalsim':
            sim_value = globalSim(vec_syna, vec_synb, type_range)
        
        token_prime = bench_data.Token_Data(word_a, word_b, sim_value)
        moretokens.append(token_prime)
        
    return(moretokens)
#similarity of two words - MaxSim

def avgSim(vecs_a, vecs_b, type_range):
    
    partial_sim = 0.0  
    
    for vec_a in vecs_a:
        for vec_b in vecs_b:
            dwab = cosine_similarity(vec_a, vec_b)
            partial_sim += dwab
            
    if not vecs_a or not vecs_b: 
        final_sim = 0.0
    else:
        final_sim = (partial_sim / (len(vecs_a) * len(vecs_b)))  
        
    final_sim = numpy.interp(final_sim, [-1, 1], RANGE[type_range]) 

    return(final_sim)           
#calculates the average similarity between two word-synsets 


def maxSim(vecs_a, vecs_b, type_range):
    high_so_far = -float('inf')
    
    for vec_a in vecs_a:
        for vec_b in vecs_b:                
            tmp_high = cosine_similarity(vec_a, vec_b)
            if(tmp_high > high_so_far):
                high_so_far = tmp_high

    new_high = numpy.interp(high_so_far, [-1, 1], RANGE[type_range]) 
  
    return(new_high)         
#calculates the highest similarity between two word-synsets      

#===============================================================================
# CONTEXT SIMILARITY
#===============================================================================
def context_sim(new_tokens, trained_model, type_range, metric='maxsimc'):
    moretokens =[]

    for new_token in new_tokens:
        word_a = new_token.word1
        word_b = new_token.word2
        #list of synset pairs
        synsets_a = sp.synset_all(word_a)
        synsets_b = sp.synset_all(word_b)
        #average vector for the context for each word
        context_a = context_parser(word_a, new_token.sent1, trained_model)
        context_b = context_parser(word_b, new_token.sent2, trained_model)
        #clean synsets that only exist in the model
        vec_syna = sp.validate_synsets_model(word_a, synsets_a, trained_model)
        vec_synb = sp.validate_synsets_model(word_b, synsets_b, trained_model)
        
        if metric == 'maxsimc':
            sim_value = maxSimC(vec_syna, context_a, vec_synb, context_b, type_range)
        elif metric == 'avgsimc':
            sim_value = avgSimC(vec_syna, context_a, vec_synb, context_b, type_range)
        elif metric == 'globalsimc':
            sim_value = globalSimC(context_a, context_b, type_range)
        
        token_prime = bench_data.Token_Data(word_a, word_b, sim_value)
        moretokens.append(token_prime)
    return(moretokens)    
#calculates the similarity of two words given a context

def context_parser(anchor_word, text_items, trained_model):
    context_vector = []
    for text_item in text_items:
        #if text_item == anchor_word: continue #discard the target/anchor word from the context - avoid bias
        synsets = sp.synset_all(text_item)
        for synset in synsets:
            key = sp.key_parser(text_item, synset)
            try:
                v1 = trained_model.word_vec(key)
                context_vector.append(v1) #put all vector words in the sentence together and average later
            except KeyError:
                pass #key not in the model
    
    return(numpy.average(context_vector, axis=0))
#give the average vector from all words in the context - discard the achor_word in the context
#change it here if weight is needed for the context

def closestSenseContext(synvecs, contextvec):
    high_so_far= -float('inf')
    nearest = []
       
    for syn in synvecs:#closest sense of 'A' to its context
        context_sim = cosine_similarity(syn, contextvec)
        if(context_sim > high_so_far):
            high_so_far = context_sim
            nearest = syn
  
    return(nearest)         
#gives the closest sense to the context attached to it

def maxSimC(vecs_a, context_a, vecs_b, context_b, type_range):
    #closet sense to context
    closest_a = closestSenseContext(vecs_a, context_a)
    closest_b = closestSenseContext(vecs_b, context_b)
    
    result = cosine_similarity(closest_a, closest_b)
    result_norm = numpy.interp(result, [-1, 1], RANGE[type_range])    
   
    return(result_norm)
#maximum similarity between v1 and v2 - takes context into account

def avgSimC(vecs_a, context_a, vecs_b, context_b, type_range):
    partial_sim = 0.0  
    
    for vec_a in vecs_a:
        pcwa = cosine_similarity(vec_a, context_a)
        for vec_b in vecs_b:
            pcwb =  cosine_similarity(vec_b, context_b)
            dwab = cosine_similarity(vec_a, vec_b)
            partial_sim += pcwa*pcwb*dwab
            
    if not vecs_a or not vecs_b: 
        final_sim = 0.0
    else:
        final_sim = (partial_sim / (len(vecs_a) * len(vecs_b)))  
        
    final_sim = numpy.interp(final_sim, [-1, 1], RANGE[type_range])
    return (final_sim)
#===============================================================================
# GLOBAL CONTEXT
#===============================================================================
def globalSim(vecs_a, vecs_b, type_range):
    
    if not vecs_a or not vecs_b:
        global_sim = 0.0
    else:    
        global_a = numpy.average(vecs_a, axis=0)
        global_b = numpy.average(vecs_b, axis=0)
        global_sim = cosine_similarity(global_a, global_b)

    global_sim = numpy.interp(global_sim, [-1, 1], RANGE[type_range])
        
    return (global_sim)
#global word vector - avg all vectors in the model and perform its similarity

def globalSimC(contex_a, context_b, type_range):
    
    global_simc = cosine_similarity(contex_a, context_b)

    global_simc = numpy.interp(global_simc, [-1, 1], RANGE[type_range])
        
    return (global_simc)

#===============================================================================
# WORDNET CONTEXT SIM
#===============================================================================
def wn_context_sim(new_tokens, trained_model, type_range, metric='maxsimc'):
    moretokens =[]

    for token in new_tokens:
        word_a = token.word1
        word_b = token.word2
        #list of synset pairs
        synsets_a = sp.synset_all(word_a)
        synsets_b = sp.synset_all(word_b)
        
        #average vector for the context for each word
        context_a = wn_context_parser(token.context1, trained_model) if token.context1 else [0.0]
        context_b = wn_context_parser(token.context2, trained_model) if token.context1 else [0.0]
        
        #clean synsets that only exist in the model
        vec_syna = sp.validate_synsets_model(word_a, synsets_a, trained_model)
        vec_synb = sp.validate_synsets_model(word_b, synsets_b, trained_model)
        
        vec_syna = [0.0] if not vec_syna else vec_syna
        vec_synb = [0.0] if not vec_synb else vec_synb
        
        if metric == 'maxsimc':
            sim_value = maxSimC(vec_syna, context_a, vec_synb, context_b, type_range)
        elif metric == 'avgsimc':
            sim_value = avgSimC(vec_syna, context_a, vec_synb, context_b, type_range)
        elif metric == 'globalsimc':
            sim_value = globalSimC(context_a, context_b, type_range)
        
        token_prime = bench_data.Token_Data(word_a, word_b, sim_value)
        moretokens.append(token_prime)
    return(moretokens)    
#calculates the similarity of two words given a context

def wn_context_parser(keys, trained_model):
    context_vector = []
    for key in keys:
        try:
            v1 = trained_model.word_vec(key)
            context_vector.append(v1) #put all vector words in the sentence together and average later
        except KeyError:
            pass #key not in the model
    
    return(numpy.average(context_vector, axis=0))
#give the average vector from all words in the context - discard the achor_word in the context


#===============================================================================
# MATH STUFF
#===============================================================================

def spearman_pearson_correlation(tokens_a, tokens_b):
    list1 = [i.simvalue for i in tokens_a ]
    list2 = [i.simvalue for i in tokens_b ]
    
    pears_v, p_rho = pearsonr(list1, list2)
    spear_v, s_rho = spearmanr(list1, list2)
    
    #print('Spearman: ', spear_v)
    #print('Pearson: ', pears_v)
    
    #print('Pearson Value \t Pearson Rho \t Spearman Value \t Spearman Rho ')
    #print('%s \t %s \t %s \t %s ' %(pears_v, p_rho, spear_v, s_rho))
    tmp_output = ('%s \t %s \t' %(pears_v, spear_v))
    return(tmp_output)
    #return(pears_v, p_rho, spear_v, s_rho)    
#calculates the spearman and pearson correlation values

def cosine_similarity(v1, v2):
    if not numpy.any(v1) or not numpy.any(v2): return(0.0) #in case there is an empty vector we return 0.0
    cos_sim = 1.0 - round(spatial.distance.cosine(v1, v2), PRECISION_COS)
    #if math.isnan(cos_dist): cos_dist = 0.0  #just to avoid NaN on the code-output for the cosine-dist value -  
    #some word vectors might be 0.0 for all dimensions
    return (cos_sim)