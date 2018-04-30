import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, render_template

from nltk import pos_tag
from nltk.corpus import wordnet as wn

lecture =[]
start_time = []
end_time = []
dic = {}
app = Flask(__name__)
@app.route('/')
def hello_world():
    global lecture
    global start_time
    global end_time
    global dic
    df_path_ret = "textretrieval_20.csv"  # PATH TO DATAFRAME HERE
    df_main_ret = pd.read_csv(df_path_ret)

    df_path_anal = "textanalysis_20.csv"  # PATH TO DATAFRAME HERE
    df_main_anal = pd.read_csv(df_path_anal)


    df_main = df_main_anal.append(df_main_ret, ignore_index = True)
    matrix = np.load("similarity.dat")
    lecture_subset = df_main['lecture'].dropna()
    start_time_subset = df_main['start_time'].dropna()
    end_time_subset = df_main['end_time'].dropna()
    indices = list(range(matrix.shape[0]))
    lecture = map(lambda x: lecture_subset[x], indices)
    start_time = map(lambda x: start_time_subset[x], indices)
    end_time = map(lambda x: end_time_subset[x], indices)

    keys = list(range(matrix.shape[0]))
    dic = {key : [] for key in keys}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if matrix[i][j] > 0.4:
                temp = dic[i]
                temp.append(j)
                dic[i] = temp
    data = {}
    for i in range(matrix.shape[0]):
        string = start_time[i]
        string += '-'
        string+= end_time[i]
        string += ' Lecture: '
        string += lecture[i]
        data[i] = string
    return render_template("home.html", dictionary = data)

@app.route('/<int:id>')
def search(id):
    sim_list = dic[id]
    data = []
    for i in range(len(sim_list)):
        index = sim_list[i]
        string = start_time[index]
        string += '-'
        string+= end_time[index]
        string += ' Lecture: '
        string += lecture[index]
        data.append(string)

    return render_template("similar.html", list = data)

def random_sentences(num_rand_sentences, df_main):
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
    lecture_subset = df_main['lecture'].dropna()
    start_time_subset = df_main['start_time'].dropna()
    end_time_subset = df_main['end_time'].dropna()

    random_tokenized_sentences = map(lambda x: tokenized_subset[x], indices)
    random_normal_sentences = map(lambda x: sentence_subset[x], indices)
    random_lecture = map(lambda x: lecture_subset[x], indices)
    random_start_time = map(lambda x: start_time_subset[x], indices)
    random_end_time = map(lambda x: end_time_subset[x], indices)

    return random_tokenized_sentences, random_normal_sentences, random_lecture, random_start_time, random_end_time

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
    """Given the word and tag return the first node in the sysnet

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


def semantic_similarity(tokenized_sentences):
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
def tfidf(tokenized_sentences):
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
def w2v(s1,s2,wordmodel):
    """Calculate similarity in Word Embeddings
    Args:
        s1 (str): a sentence
        s2 (str): a sentence
        wordmodel (wv.Word2Vec): a trained word 2 vec model

    Return:
        (float) : Similarity score
    """
    if s1==s2:
            return 1.0

    s1words=s1.split()
    s2words=s2.split()
    s1wordsset=set(s1words)
    s2wordsset=set(s2words)
    vocab = wordmodel.vocab #the vocabulary considered in the word embeddings
    if len(s1wordsset & s2wordsset)==0:
            return 0.0
    for word in s1wordsset.copy(): #remove sentence words not found in the vocab
            if (word not in vocab):
                    s1words.remove(word)
    for word in s2wordsset.copy(): #idem
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
        for j in range(i+1, len(tokenized_sentences)):
            sim = w2v(tokenized_sentences[i], tokenized_sentences[j], wordmodel)

            word2vec_mat[i][j] = sim
            word2vec_mat[j][i] = sim

    return word2vec_mat


def print_results(tokenized_sentences, sim_mat):
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
            if x!=y:
                f.write(tokenized_sentences[y] + "\n" + str(sim_mat[x][y]) + "\n")
        f.write("-"*20 + "\n")

    f.close()

if __name__ == "__main__":
    app.run()



#if __name__ == '__main__':
#    df_path = "textretrieval_20.csv"  # PATH TO DATAFRAME HERE

#    df_main = pd.read_csv(df_path)

#    tokenized_sentences, normal_sentences = random_sentences(10)

    # print tokenized_sentences

    # tfidf_mat =tfidf(tokenized_sentences)
    # print_results(normal_sentences, tfidf_mat)

    # sem_mat = semantic_similarity(tokenized_sentences)
    # print_results(normal_sentences, sem_mat)

    # wordmodel = joblib.load('textretreival_model.pkl').wv
    # w2v_mat = check_word2vec(tokenized_sentences, wordmodel)
    # print_results(normal_sentences, w2v_mat)
