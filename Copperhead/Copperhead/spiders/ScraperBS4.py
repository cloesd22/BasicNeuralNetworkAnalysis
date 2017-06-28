# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 20:52:35 2017

@author: manoj
"""

import scrapy
import re
import bs4 as bs
import urllib.request



#Scrapy
#declare class
class ScrapSniffer(scrapy.Spider):
    #declare name
    name="postSniffer"
    sauce = urllib.request.urlopen('https://www.reddit.com/r/GameDeals/comments/6jmeq3/steam_summer_sale_2017_day_5/').read()
    soup = bs.BeautifulSoup(sauce,'lxml')
    
    #declare method start_requests
    def start_requests(self):
    #declare url
        url='https://www.reddit.com/r/GameDeals/comments/6jmeq3/steam_summer_sale_2017_day_5/'
        #yield request
        yield scrapy.Request(url=url,callback=self.parse)
        
        #declare method parse
     
    def parse(self,response):

    ##==========================================================================
    ##Section pulls the data from the website using soup
    
        
        for postbody in soup.find_all('div',class_='nestedlisting'):
            
            for postblock in postbody.find_all('div',class_='thing'):
                
                #Gets post author
                try:
                    author=(postblock.find('div',class_='entry unvoted',recursive=False).find('p',class_='tagline').find('a',class_='author').string)
                   # print(author)
                except AttributeError:{}
                
                #gets post text
                try:
                    plist =postblock.find('div',class_='entry unvoted',recursive=False).find('form',class_='usertext').find('div',class_='usertext-body').find('div',class_='md').children
                                        
                    
                    try:
                        s=""
                        for pline in plist:
                            #print(pline)
                           s+=pline.string

                        #print(s)
                        
                        
                    except TypeError:{}
                        
                except AttributeError:{}

                #gets post upvotes
                try:
                    try:
                        upvote=(postblock.find('div',class_='entry unvoted',recursive=False).find('p',class_='tagline').find('span',class_='score')['title'])
                        #print(upvote)
                    except TypeError:{}
                    
                except AttributeError:{}
                
      ##========================================================================          
#            
              
                
                
#                        
                
            
              
        
        
#        for post in response.css('div.nestedlisting div.thing'):
#            
#            username = post.css('div.entry p.tagline a.author::text').extract()
#            usertext = post.css('div.entry form.usertext div.usertext-body div.md p::text').extract()
#            upvotes = post.css('div.entry p.tagline span.score::attr(title)').extract_first()
#            
#            
#            usertextstring = ' '.join(usertext)
#            usertextstringlist = re.findall("(\S+)",usertextstring)
#            wordcount = len(usertextstringlist)
#          
#            yield {'username': username,
#                   'usertext': usertext,
#                   'upvotes': upvotes,
#                   'wordcount': wordcount
#                   }
            
                    
                    #('div.nestedlisting div.thing:nth-child(1) div.entry form.usertext div.usertext-body div.md p::text').extract()
            
         #For each element
        #list element
        #IF child isn't empty, open child, 
            #pass child to THIS function, which wil list all elements, and check children again
            #if no children are found it should automatically return to the next element on the loop.
            
        
    
    #fetch each desired element of element
    #yield result.
    
    #XPath implementation, too long for any real use.
    #response.xpath('//div[contains(@class,"thing")]/div[contains(@class,
    #"entry")]/form[contains(@class,"usertext")]/div[contains(@class,"usertext-body")]
    #/div[contains(@class,"md")]')