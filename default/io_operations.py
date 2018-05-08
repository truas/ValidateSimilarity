'''
Created on Apr 27, 2018

@author: terry
'''
import os



def fname_splitter(docslist):
    fnames = []
    for doc in docslist:
        blocks = doc.split('\\')
        fnames.append(blocks[len(blocks)-1])
    return(fnames)
#getting the filenames from uri of whatever documents were processed in the input folder   

def doclist_multifolder(folder_name):
    input_file_list = []
    for roots, dir, files in os.walk(folder_name):
        for file in files:
            file_uri = os.path.join(roots, file)
            #file_uri = file_uri.replace("\\","/") #if running on windows           
            if file_uri.endswith('txt'): input_file_list.append(file_uri)
    return input_file_list
#creates list of documents in many folders

def write_ind_tokens(folder, fname, tokens): 
    #print('Saving %s Document' %bsd_fname)
    doc = open(folder +'/'+ fname, 'w+')  
    #currently using just Word \t SynsetID \t offset  \t pos
    for token in tokens:
        doc.write(token.word1 +'\t'+ token.word2 +'\t'+ str(token.simvalue) + '\n')
    doc.close()
#writes output file with word1 word2 sim_value
