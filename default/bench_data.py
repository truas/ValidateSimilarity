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
        
#class for words and their synset vectors
class Words_Data(object):
    def __init__(self):
        self.word = None
        self.semantics = None #list of 'Semantic_Data' objects

#class for semantic vector feature 
class Semantic_Data(object):
    def __init__(self):
        self.offset = None
        self.pos = None
        self.vector = None
        
class Context_Data(object):
    def __init__(self, word1, word2, sentence1, sentence2):
        self.word1 = word1
        self.word2 = word2
        self.raw1 = sentence1
        self.raw2 = sentence2
        self.ste1 = None #cleaned data for sentence 1
        self.ste2 = None #cleaned data for sentence 2