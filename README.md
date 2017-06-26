# BasicNeuralNetworkAnalysis

Basic revision exercise for simple neural network learning, data is scraped from a website and then processed through a neural network to try to arrive at predictions.

The plan:
    Use scraper to scrape a popular reddit post. Collect comments, usernames, number of
    vowels in usernames, upvotes, frequency of word usage for common words and wordcount.
    
    Put all this into a CSV
    
    Use neural networks to analyse the CSV file teaching it the links between the output (number of votes), 
    and the input, user name vowel numbers, frequency of common words used.
    
    The goal is to find some form of relationship such that number of vowels in username, word frequency and word count
    can be provided, and the upvotes of a post can be estimated atleast to a >50% rate. (ie; hopefully better than just guessing in the test data.)