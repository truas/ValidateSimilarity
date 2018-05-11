'''
Created on Apr 2, 2018

@author: Terry Ruas
'''

from default import bench_data
from default import semantic_parser as sp


def process_ws353(file):
    tokens_list = []
    print('Processing %s' %file)
    with open(file, 'r', encoding='utf-8') as fin:
        for line in fin:
            block = line.split('\t') #delimiter
            tmp_token = bench_data.Token_Data(block[0],block[1],float(block[2].strip('\n')))
            tokens_list.append(tmp_token)
    return(tokens_list)
#creates a list of ws353 data from  0.0 to 10.0

def process_rg65(file):
    tokens_list = []
    print('Processing %s' %file)
    with open(file, 'r', encoding='utf-8') as fin:
        for line in fin:
            block = line.split(';') #delimiter
            tmp_token = bench_data.Token_Data(block[0],block[1],float(block[2].strip('\n')))
            tokens_list.append(tmp_token)
    return(tokens_list)
#creates a list of rg65 data 0.0 to 4.0

def process_simlex999(file):
    tokens_list = []
    print('Processing %s' %file)
    with open(file, 'r', encoding='utf-8') as fin:
        for line in fin:
            block = line.split('\t') #delimiter
            tmp_token = bench_data.Token_Data(block[0], block[1], float(block[3].strip('\n')))
            tokens_list.append(tmp_token)
    return(tokens_list)
#creates a list of simlex999 linear mapped from 0 to 6 -> 0.0 to 10.0 - only same pos are comapred

def process_men(file):
    tokens_list = []
    print('Processing %s' %file)
    with open(file, 'r', encoding='utf-8') as fin:
        for line in fin:
            block = line.split(' ') #delimiter
            tmp_token = bench_data.Token_Data(block[0], block[1], float(block[2].strip('\n')))
            tokens_list.append(tmp_token)
    return(tokens_list)
#creates a list of MEN linear mapped from 1 to 7

def process_mc28(file):
    tokens_list = []
    print('Processing %s' %file)
    with open(file, 'r', encoding='utf-8') as fin:
        for line in fin:
            block = line.split(';') #delimiter
            tmp_token = bench_data.Token_Data(block[0], block[1], float(block[2].strip('\n')))
            tokens_list.append(tmp_token)
    return(tokens_list)
#creates a list of MC28 linear mapped from 0 to 4

def process_yp130(file):
    tokens_list = []
    print('Processing %s' %file)
    with open(file, 'r', encoding='utf-8') as fin:
        for line in fin:
            block = line.split(' ') #delimiter
            tmp_token = bench_data.Token_Data(block[0], block[1], float(block[2].strip('\n')))
            tokens_list.append(tmp_token)
    return(tokens_list)
#creates a list of YP130 linear mapped from 0 to 4

def process_simverb(file):
    tokens_list = []
    print('Processing %s' %file)
    with open(file, 'r', encoding='utf-8') as fin:
        for line in fin:
            block = line.split('\t') #delimiter
            tmp_token = bench_data.Token_Data(block[0], block[1], float(block[3].strip('\n')))
            tokens_list.append(tmp_token)
    return(tokens_list)
#creates a list of SimVerb-3500 linear mapped from 0 to 10

def process_stanford(file):
    tokens_list = []
    print('Processing %s' %file)
    with open(file, 'r', encoding='utf-8') as fin:
        for line in fin:
            block = line.split('\t') #delimiter
            tmp_token = bench_data.Stanford_Data(block[1], block[2], block[3], block[4], block[5], block[6], float(block[7].strip('\n')))
            tokens_list.append(tmp_token)
    return(tokens_list)
#creates a list of Stanford linear mapped from 0.0 to 10.0

def sentece_wrapper(file):
    tokens_list = []
    print('Processing %s' %file)
    with open(file, 'r', encoding='utf-8') as fin:
        for line in fin:
            block = line.split('\t') #delimiter
            work_token = bench_data.Work_Data()
            work_token.word1 = block[1]
            work_token.word2 = block[3]
            work_token.sent1 = sp.tokenize_text(block[5])
            work_token.sent2 = sp.tokenize_text(block[6])
            work_token.simvalue = float(block[7].strip('\n'))
            tokens_list.append(work_token)
    return(tokens_list)
#reads and wrap tokens separated  by <tab>
#creates a list of Stanford linear mapped from 0.0 to 10.0





 