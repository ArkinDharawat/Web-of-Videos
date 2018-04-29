import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sematch.semantic.similarity import WordNetSimilarity

WNS = WordNetSimilarity()


# NOTE: For reference see: https://pdfs.semanticscholar.org/1374/617e135eaa772e52c9a2e8253f49483676d6.pdf

def random_sentences(num_rand_sentences):
    """Select num_rand_sentences at random from the Dataframe

    Args:
        num_rand_sentences (int): the number of sentences to select at random

    Return:
         list: list of sentences
    """
    size = num_rand_sentences
    indices = np.random.randint(0, df_main.shape[0], size)

    tokenized_subset = df_main['tokenized_sentence'].dropna()
    sentence_subset = df_main['sentence'].dropna()

    random_tokenized_sentences = map(lambda x: tokenized_subset[x], indices)
    random_normal_sentences = map(lambda x: sentence_subset[x], indices)

    return random_tokenized_sentences, random_normal_sentences


## Semantic Similarity

SIMILARITY = "wup"  # NOTE: Using Wup, closest to the Combined metric.


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


def maximum_similarity(word, tokens):
    """ Calculate maximum similarity from word and token

    Args:
        word (str): the word
        tokens (list): tokens belonging to the term

    Return:
         float: the maxSim score
    """
    if len(tokens) == 0:
        return 0

    best_score = max([WNS.word_similarity(word, token, SIMILARITY) for token in tokens])
    if best_score is None:
        return 0
    else:
        return best_score


def similarity_measure(tokens1, tokens2, idf):
    """Calulcate the similarity metric between 2 terms

    Args:
        tokens1 (list): list of tokens for term 1
        tokens2 (list): list of tokens for term 2
        idf (dict): the idf dictionary

    Return:
        (float) the similarity score
    """
    try:

        sum_idf_1 = sum(map(lambda x: idf[x], tokens1))
        sum_idf_2 = sum(map(lambda x: idf[x], tokens2))
        max_sim_1 = 0.0
        max_sim_2 = 0.0

        for i in range(len(tokens1)):
            score = maximum_similarity(tokens1[i], tokens2) * idf[tokens1[i]]
            max_sim_1 += score

        for i in range(len(tokens2)):
            score = maximum_similarity(tokens2[i], tokens1) * idf[tokens2[i]]
            max_sim_2 += score

        return 0.5 * (float(max_sim_1) / float(sum_idf_1) + float(max_sim_2) / float(sum_idf_2))
    except:
        print tokens1, tokens2
        return -1


def semantic_similarity(tokenized_sentences):
    """Make a matrix of semantic similarity between i and j entries

    Args:
        tokenized_sentences (list): list of tokens of sentences

    Return:
        np.narray: the matrix with the similarty scores
    """
    token_list = map(lambda x: x.split(" ")[1:], tokenized_sentences)

    word_idf = idf(token_list)  # TODO: Replace with sklearn?

    sentence_mat = np.ones(shape=(len(tokenized_sentences), len(tokenized_sentences)))

    for i in range(len(tokenized_sentences)):
        for j in range(i + 1, len(tokenized_sentences)):
            sim = round(similarity_measure(token_list[i], token_list[j], word_idf), 3)

            sentence_mat[i][j] = sim
            sentence_mat[j][i] = sim  # TODO: This measures for words with same part of speech

            # print sim

            # break
    return sentence_mat


##  TF-IDF
def tfidf(tokenized_sentences):
    """Make a matrix of tfidf similarity between i and j entries

    Args:
        tokenized_sentences (list): list of tokens of sentences


    Return:
        np.narray: the matrix with the tfidf scores
    """
    tfidf_vectorizer = TfidfVectorizer(min_df=1)
    tfidf_vect = tfidf_vectorizer.fit_transform(
        tokenized_sentences)  # each sentence can be replaced by a whole document
    tfidf_mat = (tfidf_vect * tfidf_vect.T).A  # similarities matrix
    for i in range(len(tfidf_mat)):
        for j in range(len(tfidf_mat)):
            tfidf_mat[i][j] = round(tfidf_mat[i][j], 3)

    return tfidf_mat


## Word Embeddings
def w2v(s1, s2, wordmodel):
    """Calculate similarity in Word Embeddings
    Args:
        s1 (str): a sentence
        s2 (str): a sentence
        wordmodel (wv.Word2Vec): a trained word 2 vec model

    Return:
        (float) : Similarity score
    """
    if s1 == s2:
        return 1.0

    s1words = s1.split()
    s2words = s2.split()
    s1wordsset = set(s1words)
    s2wordsset = set(s2words)
    vocab = wordmodel.vocab  # the vocabulary considered in the word embeddings
    if len(s1wordsset & s2wordsset) == 0:
        return 0.0
    for word in s1wordsset.copy():  # remove sentence words not found in the vocab
        if (word not in vocab):
            s1words.remove(word)
    for word in s2wordsset.copy():  # idem
        if (word not in vocab):
            s2words.remove(word)
    return wordmodel.n_similarity(s1words, s2words)


def word2vec(tokenized_sentences, wordmodel):
    """Calculate similarit matrix using a word 2 vec model

    Args:
        tokenized_sentences (list): a list of tokenized sentence
        wordmodel (wv.Word2Vec): a word to vec model

    Return:
         (np.ndarray): a similarity matrix
    """
    word2vec_mat = np.zeros(shape=(len(tokenized_sentences), len(tokenized_sentences)))

    for i in range(len(tokenized_sentences)):
        for j in range(i + 1, len(tokenized_sentences)):
            sim = w2v(tokenized_sentences[i], tokenized_sentences[j], wordmodel)

            word2vec_mat[i][j] = sim
            word2vec_mat[j][i] = sim

    return word2vec_mat


def write_to_results(tokenized_sentences, sim_mat):
    """Print results from the matrix generated

    Args:
        tokenized_sentences (list): list of sentences
        sim_mat (np.narray): the matrix with similarity scores

    Return:
        (void)
    """

    f = open("results.txt", "w")

    for x in range(len(sim_mat)):
        f.write(tokenized_sentences[x] + "\n")
        for y in range(len(sim_mat[0])):
            if x != y:
                f.write(tokenized_sentences[y] + "\n" + str(sim_mat[x][y]) + "\n")
        f.write("-" * 20 + "\n")

    f.close()


if __name__ == '__main__':
    df_path = "textretrieval_20.csv"  # PATH TO DATAFRAME HERE

    df_main = pd.read_csv(df_path)

    tokenized_sentences, normal_sentences = random_sentences(2)

    # print tokenized_sentences

    # tfidf_mat =tfidf(tokenized_sentences)
    # print_results(normal_sentences, tfidf_mat)

    # sem_mat = semantic_similarity(tokenized_sentences)
    # print_results(normal_sentences, sem_mat)

    # wordmodel = joblib.load('textretreival_model.pkl').wv
    # w2v_mat = check_word2vec(tokenized_sentences, wordmodel)
    # print_results(normal_sentences, w2v_mat)
