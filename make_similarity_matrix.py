import pandas as pd
import similarity_measures
import sys

def main(df_path1, df_path2):
    if df_path1 == "":
        df_path1 = "textanalysis_20.csv"
    df_main_ta = pd.read_csv(df_path1)
    s_ta = df_main_ta['tokenized_sentence'].dropna()

    if df_path2 == "":
        df_path2 = "textretrieval_20.csv"
    df_main_tr = pd.read_csv(df_path2)
    s_tr = df_main_tr['tokenized_sentence'].dropna()

    s = list(s_ta) + list(s_tr)
    print "Total sentences {0} and Total similarity scores {1} ".format(len(s), len(s)*len(s))

    test_sentences = s
    tfidf_mat = similarity_measures.tfidf(test_sentences)
    print "DONE"
    sem_mat = similarity_measures.semantic_similarity(test_sentences)


    sim_mat = 0.5 * (tfidf_mat + sem_mat)
    sim_mat.dump("similarity.dat")

if __name__ == '__main__':
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main("", "")


