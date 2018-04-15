import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import pos_tag
from nltk.corpus import wordnet as wn

import sys
from gensim.models import word2vec, KeyedVectors
#TODO: Make a class or keep them as functions????

def random_sentences(num_rand_sentences):
    """Select num_rand_sentences at random from the Dataframe

    Args:
        num_rand_sentences (int): the number of sentences to select at random

    Return:
         list: list of sentences
    """
    size = num_rand_sentences
    indices = np.random.randint(0, df_main.shape[0], size)
    random_sentences = map(lambda x: df_main.iloc[x]['tokenized_sentence'], indices)

    return random_sentences

## Semantic Similarity

SIMILARITY = 0

def idf(tokenized_sentences):
        """Calculate of given list of sentences

        Args:
            tokenized_sentences (list): list of tokenized sentences

        Return:
            dict: a dictioanry of idf values
        """
        idf_values = {}
        all_tokens_set = set([item for sublist in tokenized_sentences for item in sublist])

        for tkn in all_tokens_set:
            contains_token = map(lambda doc: tkn in doc, tokenized_sentences)
            idf_values[tkn] = 1 + np.log(len(tokenized_sentences) / (sum(contains_token)))
        return idf_values


def penn_to_wn(tag):
    """Convert between a Penn Treebank tag to a simplified Wordnet tag

    Args:
        tag (): the POS tag assigned

    Return:
        char: character denoting the POS tag in Wordnet
    """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    """ Given the word and tag return the first node in the sysnet

    Args:
        word (str): the word
        tag (): the POS tag assigned

    Return:
        wn.synet: the root of the synet of the word
    """
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def maximum_similarity(word, tokens, synet_word, synet_token):
    """ Calculate maximum similarity from word and token
    Args:
        word (str): the word
        tokens (list): tokens belonging to the term
        synet_word (wn.synet): the synet of the word
        synet_token (list): the list of synets

    Return:
         float: the maxSim score
    """
    if SIMILARITY == 0:
         best_score =  max([synet_word.path_similarity(synet) for synet in synet_token])
         if best_score is None:
             return 0
         else:
             return best_score

def similarity_measure(tokens1, tokens2, synet1, synet2, idf):
    """Calulcate the similarity metric between 2 terms

    Args:
        tokens1 (list): list of tokens for term 1
        tokens2 (list): list of tokens for term 2
        synet1 (list): list of synets of term 1
        synet2 (list): list of synets of term 2
        idf (dict): the idf dictionary

    Return:
        (float) the similarity score
    """

    sum_idf_1 = sum(map(lambda x: idf[x], tokens1))
    sum_idf_2 = sum(map(lambda x: idf[x], tokens2))
    max_sim_1 = 0.0
    max_sim_2 = 0.0

    for i in range(len(tokens1)):
        if synet1[i] is None:
            score = 0
        else:
            score = maximum_similarity(tokens1[i], tokens2, synet1[i], [x for x in synet2 if x]) * idf[tokens1[i]]

        max_sim_1+=score

    for i in range(len(tokens2)):
        if synet2[i] is None:
            score = 0
        else:
            score = maximum_similarity(tokens2[i], tokens1, synet2[i], [x for x in synet1 if x]) * idf[tokens2[i]]
        max_sim_2+=score

    return 0.5 * ( float(max_sim_1)/float(sum_idf_1) + float(max_sim_2)/float(sum_idf_2))


def check_semantic_similarity(tokenized_sentences):
    """Make a matrix of semantic similarity between i and j entries

    Args:
        tokenized_sentences (list): list of tokens of sentences

    Return:
        np.narray: the matrix with the similarty scores
    """
    token_list = map(lambda x: x.split(" ")[1:], tokenized_sentences)
    synet_list = map(lambda x: [tagged_to_synset(*y) for y in pos_tag(x)], token_list)


    word_idf = idf(token_list)

    sentence_mat = np.zeros(shape=(len(tokenized_sentences), len(tokenized_sentences)))

    for i in range(len(tokenized_sentences)):
        for j in range(i+1, len(tokenized_sentences)):
            sim = similarity_measure(token_list[i], token_list[j], synet_list[i], synet_list[j], word_idf)

            sentence_mat[i][j] = sim
            sentence_mat[j][i] = sim #TODO: Path Similarity is not commutative! 

            # print sim

            # break
    return sentence_mat

##  TF-IDF

def check_tfidf(tokenized_sentences):
    """Make a matrix of tfidf similarity between i and j entries
    Args:
        tokenized_sentences (list): list of tokens of sentences

    Return:
        np.narray: the matrix with the tfidf scores
    """
    tfidf_vectorizer = TfidfVectorizer(min_df=1)
    tfidf_vect = tfidf_vectorizer.fit_transform(tokenized_sentences) #each sentence can be replaced by a whole document
    tfidf_mat = (tfidf_vect * tfidf_vect.T).A #similarities matrix

    return tfidf_mat

## Word Embeddings
# TODO: WORK ON IMPROVING WORD EMBEDDINGS
# def w2v(s1,s2,wordmodel):
#         if s1==s2:
#                 return 1.0
#
#         s1words=s1.split()
#         s2words=s2.split()
#         s1wordsset=set(s1words)
#         s2wordsset=set(s2words)
#         vocab = wordmodel.vocab #the vocabulary considered in the word embeddings
#         if len(s1wordsset & s2wordsset)==0:
#                 return 0.0
#         for word in s1wordsset.copy(): #remove sentence words not found in the vocab
#                 if (word not in vocab):
#                         s1words.remove(word)
#         for word in s2wordsset.copy(): #idem
#                 if (word not in vocab):
#                         s2words.remove(word)
#         return wordmodel.n_similarity(s1words, s2words)
#
# def check_word2vec(tokenized_sentences, wordmodel):
#     word2vec_mat = np.zeros(shape=(len(tokenized_sentences), len(tokenized_sentences)))
#
#     for i in range(len(tokenized_sentences)):
#         for j in range(i+1, len(tokenized_sentences)):
#             sim = w2v(tokenized_sentences[i][j], tokenized_sentences[j][i], wordmodel)
#
#             word2vec_mat[i][j] = sim
#             word2vec_mat[j][i] = sim
#
#     return word2vec_mat


def print_results(tokenized_sentences, test_index, sim_mat):
    """Print results from the matrix generated

    Args:
        tokenized_sentences (list): list of sentences
        test_index (int): the index to print
        sim_mat (np.narray): the matrix with similarity scores

    Return:
        (void)
    """
    if test_index > len(tokenized_sentences):
        test_index = 0

    print tokenized_sentences[test_index]
    print

    for x in range(len(tokenized_sentences)):
        if x!=test_index:
            print tokenized_sentences[x], sim_mat[test_index][x]


if __name__ == '__main__':


    df_path = "textanalysis_10.csv"  # PATH TO DATAFRAME HERE

    df_main = pd.read_csv(df_path)

    tokenized_sentences = random_sentences(10)

    # sem_mat = check_semantic_similarity(tokenized_sentences)
    # tfidf_mat =check_tfidf(tokenized_sentences)
    #
    # print_results(tokenized_sentences, 4, sem_mat)

    # wordmodelfile = "GoogleNews-vectors-negative300.bin.gz"
    # wordmodel = KeyedVectors.load_word2vec_format(wordmodelfile, binary=True)
    #
    # w2v_mat = check_word2vec(tokenized_sentences, wordmodel)
    #
    # print_results(tokenized_sentences, 4, w2v_mat)

