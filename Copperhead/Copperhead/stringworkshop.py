# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 04:23:59 2017

@author: manoj
"""

##String Processor, capable of doing advanced mass string functions such as
#word count, word frequency calculation etc.


class stringworkshop():
    
    def wordcount(text):
        #simple word count, input text, returns integer wordcount
        wordcount = text.split()
        wordcount = (len(wordcount))
        return wordcount
    
    def wordfreq(word,text):
        #simple word frequency, provide text, and word returns frequency integer.
        count = 0
        
        textarray = text.split()
        
        for unit in textarray:
            if word ==unit:
                count=count+1
        
        return count
        
        