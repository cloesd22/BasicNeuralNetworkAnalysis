# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 20:52:35 2017

@author: manoj
"""

import scrapy


#Scrapy
#declare class
class scrapy(scrapy.Spider):
    #declare name
    name="postSniffer"
    
    #declare method start_requests
    def start_requests(self):
    #declare url
        url='https://www.reddit.com/r/GameDeals/comments/6jmeq3/steam_summer_sale_2017_day_5/'
        #yield request
        yield scrapy.request(url=url,callback=self.parse)
        
        #declare method parse
     
    def parse(self,response):
        
        for post in response.css('div.nestedlisting div.thing'):
            yield{
                    'usertext': post.css('div.nestedlisting div.thing div.entry form.usertext div.usertext-body div.md p::text').extract()
                    
                }
            
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