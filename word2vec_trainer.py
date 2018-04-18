from gensim.models import Word2Vec
import pandas as pd
from sklearn.externals import joblib


def csv_to_train(csv):
    """CSV to train the data one

    Args:
        csv (pd.Dataframe):

    Return
        (void):
    """
    df_main = pd.read_csv(csv)
    sentences = map(lambda x: x.split(" "), df_main['tokenized_sentence'].dropna())[1:] # Seems like the correct way to break stuff up
    # print sentences
    model = Word2Vec(sentences)
    joblib.dump(model, 'textretreival_model.pkl')


csv_to_train("textretrieval.csv")