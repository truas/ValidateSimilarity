'''
Created on Apr 2, 2018

@author: Terry Ruas
'''
#import
import logging
import gensim
import time
import nltk
import sys
import argparse #for command line arguments
import os

#from-imports
from default import io_operations as io
from default import sim_calc as sc
from stop_words import get_stop_words

#folders-files
in_foname = 'C:/tmp_project/ValidateSimilarity/input_ws'
ou_foname = 'C:/tmp_project/ValidateSimilarity/output'
#mo_foname = 'C:/Users/terry/Documents/Datasets/Wikipedia_Dump/2018_01_20/models/dbsd/500d-neg-10w-2mc-sg/wikidump20180120-500d-neg-10w-2mc-sg.vector'
mo_foname = 'C:/Users/terry/Documents/Datasets/Wikipedia_Dump/2010_04_08/models/refine/300d-05w-05w-hs-sg-rf.model'

range_category = {'cos': 0, 'ws353': 1, 'simlex': 1,'stanford': 1, 'simverb': 1, 'rg65': 2, 'mc28': 2, 'yp130': 2, 'men': 3}
                      #0        #1        #2        
metric_category = ['avgsim', 'maxsim', 'globalsim'] #maxsimc = localsim
MC = metric_category[2]

#python module absolute path
pydir_name = os.path.dirname(os.path.abspath(__file__))

#python path definition
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
nltk.download('wordnet') #just to guarantee wordnet from nltk is installed

#overall runtime start
start_time = time.monotonic()

#Main core
if __name__ == '__main__':  
    
    #show logs
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
     
#===============================================================================
#     #IF you want to use COMMAND LINE for folder path
#     parser = argparse.ArgumentParser(description="BSD_Extractor - Transforms text into synsets")
#     parser.add_argument('--input', type=str, action='store', dest='inf', metavar='<folder>', required=True, help='input folder to read document(s)')
#     parser.add_argument('--output', type=str, action='store', dest='ouf', metavar='<folder>', required=True, help='output folder to write document(s)')
#     parser.add_argument('--model', type=str, action='store', dest='mod', metavar='<folder>', required=True, help='trained word embeddings model')
#     
#     args = parser.parse_args()
#      
#     #COMMAND LINE  folder paths
#     input_folder = args.inf
#     output_folder = args.ouf
#     model_folder = args.mod
# 
#     #in/ou relative location - #input/output/model folders are under synset/module/
#     in_foname = os.path.join(pydir_name, '../'+input_folder) 
#     ou_foname = os.path.join(pydir_name, '../'+output_folder)
#     mo_foname = os.path.join(pydir_name, '../'+model_folder)
#===============================================================================
    
    #Loads
    #trained_w2v_model = gensim.models.KeyedVectors.load_word2vec_format(mo_foname, binary=False) #If the model is not binary set binary=False
    trained_w2v_model = gensim.models.KeyedVectors.load(mo_foname) #model.load used with .model extension - this files has to be in the same folder as its .npy
    results = ""
    
    docs = io.doclist_multifolder(in_foname)#creates list of documents to parse
    docsnames = io.fname_splitter(docs) #name of the document
    counter = 0 #just to control the output file name
   
   #============================================================================
   # MEN - 0
   #============================================================================
    tokens = io.process_men(docs[0])
    new_tokens = sc.nocontext_sim(tokens, trained_w2v_model, range_category['men'], MC)
    #io.write_ind_tokens(ou_foname, docsnames[0], new_tokens)
    results += sc.spearman_pearson_correlation(tokens, new_tokens)
    
   #============================================================================
   # RG65 - 1
   #============================================================================
    tokens = io.process_rg65(docs[1])
    new_tokens = sc.nocontext_sim(tokens, trained_w2v_model, range_category['rg65'], MC)
    #io.write_ind_tokens(ou_foname, docsnames[1], new_tokens)
    results += sc.spearman_pearson_correlation(tokens, new_tokens)
   
   #============================================================================
   #SIMLEX999 - 2
   #============================================================================
    tokens = io.process_simlex999(docs[2])
    new_tokens = sc.nocontext_sim(tokens, trained_w2v_model, range_category['simlex'], MC)
    #io.write_ind_tokens(ou_foname, docsnames[2], new_tokens)
    results += sc.spearman_pearson_correlation(tokens, new_tokens)
    
   #============================================================================
   #wsim353 - 3
   #============================================================================
    tokens = io.process_ws353(docs[3])
    new_tokens = sc.nocontext_sim(tokens, trained_w2v_model, range_category['ws353'], MC)
    #io.write_ind_tokens(ou_foname, docsnames[3], new_tokens)
    results += sc.spearman_pearson_correlation(tokens, new_tokens)
    
   #============================================================================
   #Stanford - 4
   #============================================================================
    tokens = io.process_stanford(docs[4])
    new_tokens = sc.nocontext_sim(tokens, trained_w2v_model, range_category['stanford'], MC)
    #io.write_ind_tokens(ou_foname, docsnames[4], new_tokens)
    results += sc.spearman_pearson_correlation(tokens, new_tokens)
    
   #============================================================================
   #MC-28 - 5
   #============================================================================
    tokens = io.process_mc28(docs[5])
    new_tokens = sc.nocontext_sim(tokens, trained_w2v_model, range_category['mc28'], MC)
    #io.write_ind_tokens(ou_foname, docsnames[5], new_tokens)
    results += sc.spearman_pearson_correlation(tokens, new_tokens)  
              
   #============================================================================
   #YP-130 - 6
   #============================================================================
    tokens = io.process_yp130(docs[6])
    new_tokens = sc.nocontext_sim(tokens, trained_w2v_model, range_category['yp130'], MC)
    #io.write_ind_tokens(ou_foname, docsnames[6], new_tokens)
    results += sc.spearman_pearson_correlation(tokens, new_tokens)
    
   #============================================================================
   #SimVerb-3500 - 7
   #============================================================================
    tokens = io.process_simverb(docs[7])
    new_tokens = sc.nocontext_sim(tokens, trained_w2v_model, range_category['simverb'], MC)
    #io.write_ind_tokens(ou_foname, docsnames[7], new_tokens)
    results += sc.spearman_pearson_correlation(tokens, new_tokens)    
     
    print('\n', results, '\n') #Print results for WS benchmark - divided by 'tab' and in order of execution p-value, p-rho,s-value, s-rho
         
    print('finished...')