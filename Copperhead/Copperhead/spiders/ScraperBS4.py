# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 20:52:35 2017

@author: manoj
"""

import sys
sys.path.insert(0, "E:\SnakeGarden\CustomClasses\stringworkshop")
sys.path.insert(0, "E:\SnakeGarden\Copperhead\Copperhead\Copperhead")
import pprint
import scrapy
import re
import bs4 as bs
import urllib.request
import stringworkshop
from items import comment



#Scrapy
#declare class
class ScrapSniffer(scrapy.Spider):

    name="postSniffer"

    #declare method start_requests
    def start_requests(self):
    #declare url
                #yield request
        url = 'https://www.reddit.com/r/GameDeals/comments/6jmeq3/steam_summer_sale_2017_day_5/'
        yield scrapy.Request(url=url,callback=self.parse)
        

    def parse(self,response):
    ##==========================================================================
    ##Section pulls the data from the website using 
        from stringworkshop import stringworkshop as sws
        urlsoup = 'https://www.reddit.com/r/GameDeals/comments/6jmeq3/steam_summer_sale_2017_day_5/'
        sauce = urllib.request.urlopen(urlsoup).read()
        soup = bs.BeautifulSoup(sauce,'lxml')
        for postbody in soup.find_all('div',class_='nestedlisting'):
            
           # for postblock in postbody.find_all('div',class_='thing comment'):
             for postblock in postbody.select('div.thing.comment'):   
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
                #Declares metrics                
                wordcount = sws.wordcount(s)
                thecount = sws.wordfreq("the",s)
                Icount = sws.wordfreq("I",s)
                andcount = sws.wordfreq("and",s)
                avgwlength = sws.avgwordlength(s)
                
                
                commentexport = comment(
                        
                                author=scrapy.Field(),
                                usertext=scrapy.Field(),
                                thecount=scrapy.Field(),
                                icount=scrapy.Field(),
                                andcount=scrapy.Field(),
                                avgwl=scrapy.Field(),
                                upvote=scrapy.Field()
                        
                                )
                
                yield{'author':author,
                       'usertext':s,
                       'wordcount':wordcount,
                       'thecount':thecount,
                       'Icount':Icount,
                       'andcount':andcount,
                       'avgwl':avgwlength,
                       'upvotes':upvote
                        }
                        
                
      ##========================================================================          


                
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