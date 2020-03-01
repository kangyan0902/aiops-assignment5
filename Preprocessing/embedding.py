
import os
import zipfile

class embedding():
    def __init__(self,max_length_dictionary):
        self.max_length_dictionary=max_length_dictionary
        module_path = os.path.abspath(__file__)
        self.dictionary_path = os.path.join(os.path.dirname(module_path), "..","resources", "glove.twitter_aicloud.27B.25d.txt")
        
    def load_embedding_dictionary(self,dictionary_path):


        self.embedding_dictionary = {}

        embeddings = []

        if ".zip/" in dictionary_path:
            archive_path = os.path.abspath(dictionary_path)

            split = archive_path.split(".zip/")

            archive_path = split[0] + ".zip/"
            path_inside = split[0]

            archive = zipfile.ZipFile(archive_path, "r")
            embeddings = archive.read(path_inside).decode("utf8").split("\n")

        else:

            embeddings = open(dictionary_path, "r", encoding="utf8").read().split("\n")

        for index, row in enumerate(embeddings):

            split = row.split(" ")

            if index == self.max_length_dictionary:
                return 

            self.embedding_dictionary[split[0]] = index

    def replace_token_with_index(self, text):
        """
     This function is used for replacing tokens with indices
     """
        tokened = [self.embedding_dictionary[word] for word in text]
        return [tokened]
       