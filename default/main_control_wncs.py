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
from default import semantic_parser as sp
from default import sim_calc as sc
from datetime import timedelta
from stop_words import get_stop_words

#folders-files
in_foname = 'C:/tmp_project/ValidateSimilarity/input_cs'
ou_foname = 'C:/tmp_project/ValidateSimilarity/output'
mo_w2v = 'C:/Users/terry/Documents/Datasets/GoogleNews/GoogleNews-vectors-negative300.bin'
mo_s2v = 'C:/Users/terry/Documents/Datasets/Wikipedia_Dump/2010_04_08/models/refine/300d-15w-10mc-hs-cbow-rf.model'


range_category = {'cos': 0, 'ws353': 1, 'simlex': 1,'stanford': 1, 'simverb': 1, 'rg65': 2, 'mc28': 2, 'yp130': 2, 'men': 3}
                      #0        #1        #2         
metric_category = ['avgsimc', 'maxsimc', 'globalsimc'] #maxsimc = localsim
MC = metric_category[0]

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
    #w2v_model = gensim.models.KeyedVectors.load_word2vec_format(mo_w2v, binary=True) #in case word2vec is provided
    s2v_model = gensim.models.KeyedVectors.load(mo_s2v) #model.load used with .model extension - this files has to be in the same folder as its .npy
    refi_flag = True
    results = ""
    
    docs = io.doclist_multifolder(in_foname)#creates list of documents to parse
    docsnames = io.fname_splitter(docs) #name of the document
    counter = 0 #just to control the output file name
   
   #============================================================================
   #WordSimilarity with Context
   #============================================================================
    tokens = io.sentece_wrapper(docs[0])
    wn_tokens_processed = sp.wn_sentence_handler(tokens, s2v_model, refi_flag)
    better_tokens = sc.wn_context_sim(wn_tokens_processed,s2v_model, range_category['stanford'], MC)
    results += sc.spearman_pearson_correlation(tokens, better_tokens)

   
    print('\n', results, '\n') #Print results for WS benchmark - divided by 'tab' and in order of execution p-value, p-rho,s-value, s-rho
         
    print('finished...')