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
from default import process_synsets as pr
from default import semantic_parser as sp
from default import sim_calc as sc
from datetime import timedelta
from stop_words import get_stop_words

#folders-files
in_foname = 'C:/tmp_project/ValidateSimilarity/input_cs'
ou_foname = 'C:/tmp_project/ValidateSimilarity/output'
#mo_foname = 'C:/Users/terry/Documents/Datasets/Wikipedia_Dump/2018_01_20/models/dbsd/500d-neg-10w-2mc-sg/wikidump20180120-500d-neg-10w-2mc-sg.vector'
mo_foname = 'C:/Users/terry/Documents/Datasets/Wikipedia_Dump/2010_04_08/models/300d-hs-15w-10mc-cbow.model'

range_category = {'cos': 0, 'ws353': 1, 'simlex': 1,'stanford': 1, 'simverb': 1, 'rg65': 2, 'mc28': 2, 'yp130': 2, 'men': 3}
                      #0        #1        #2         
metric_category = ['avgsimc', 'maxsimc', 'globalsimc'] #maxsimc = localsim
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
   #Stanford - 4
   #============================================================================
    tokens = pr.process_stanford(docs[0])
    new_tokens = sp.sentence_adapter(tokens)
    better_tokens = sc.context_sim(new_tokens, trained_w2v_model, range_category['stanford'], MC)
    results += sc.spearman_pearson_correlation(tokens, better_tokens)

    #new_tokens = sc.synset_similarity(tokens, trained_w2v_model, range_category['stanford'])
    #io.write_ind_tokens(ou_foname, docsnames[0], new_tokens)
    
    
   
    print('\n', results, '\n') #Print results for WS benchmark - divided by 'tab' and in order of execution p-value, p-rho,s-value, s-rho
         
    print('finished...')