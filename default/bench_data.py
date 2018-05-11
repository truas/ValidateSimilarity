'''
Created on Apr 2, 2018

@author: admin
'''
#class for WS353, RG65, SimLex999, MEN dataset
class Token_Data(object):
    def __init__(self, word1, word2, simvalue):
        self.word1 = word1
        self.word2 = word2
        self.simvalue = simvalue
        
#class for Stanford dataset
class Stanford_Data(object):
    def __init__(self, word1, pos1, word2, pos2, sent1, sent2, simvalue):
        self.word1 = word1
        self.pos1 = pos1
        self.word2 = word2
        self.pos2 = pos2
        self.sent1 = sent1 #target word  between <b> </b>
        self.sent2 = sent2 #target word  between <b> </b>
        self.simvalue = simvalue
        
#class for Generic dataset with context  
class Work_Data(object):
    def __init__(self):
        self.word1 = None
        self.pos1 = None
        self.word2 = None
        self.pos2 = None
        self.sent1 = None #target word  between <b> </b>
        self.sent2 = None #target word  between <b> </b>
        self.simvalue = None
                     
class WN_Token_Data(object):
    def __init__(self, word1, word2):
        self.word1 = word1
        self.word2 = word2
        self.context1 = None
        self.context2 = None
        self.simvalue = None        
        
#===============================================================================
# Data Related to WordNet Approach
#===============================================================================

class WNData(object): 
    def __init__(self, sys_id, offset, pos, gloss):
        self.sys_id = sys_id
        self.offset = offset
        self.pos = pos
        self.gloss = gloss
        self.gloss_avg_vec = list()
        self.vector = [] #in case embedding is using word-offset-pos
#WordNet object for sentence

#'WordsData' object for word in a 'Document'
class WordsData(object):
    def __init__(self, word):
        self.word = word #word itself
        self.wndatapack = None #list of WNData items
        self.prime_sys = None #the 'PrimeData' containing SynsetID, Offset and POS for later retrieval   

#prime 'SynsetData'(prime_sys) in  'WordsData'   
class PrimeData(object):
    def __init__(self, psys_id, poffset, ppos):
        self.psys_id = psys_id
        self.poffset = poffset
        self.ppos = ppos