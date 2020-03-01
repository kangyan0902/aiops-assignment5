import re
import csv
from nltk import tokenize
import os
import zipfile
from Preprocessing.embedding import embedding

class Preprocess():
    """
 class
 """
    def __init__(self, max_length_dictionary=5000000, max_length_tweet=20):
        """
     initiate instance
     """
        self.max_length_tweet = max_length_tweet
        self.embedding=embedding(max_length_dictionary)
        self.embedding.load_embedding_dictionary(self.embedding.dictionary_path)
   

    def clean_text(self,text):
        """
        This function is used for clean raw strings
    """
        text = re.sub(r'http://\S+.\S+', '', text)
        text = text.lower()
        return text

    def tokenize_text(self,text):
        """
        This function is used for tokenizing text
     """
        tknzr = tokenize.TweetTokenizer(reduce_len=True)
        return tknzr.tokenize(text)


    def pad_sequence(self,text):
        """
     This function is used for create pad sequence
     """
        for element in text:
            if len(element) > self.max_length_tweet:
                element = element[:self.max_length_tweet]
            else:
                element = element.extend([0] * (self.max_length_tweet - len(element)))
        return text

    def pre_process_text(self,text):

        cleaned_text=self.clean_text(text)
        tokens=self.tokenize_text(cleaned_text)
        embedding_indexes=self.embedding.replace_token_with_index(tokens)
        indextext=self.pad_sequence(embedding_indexes)
        return indextext



    


    