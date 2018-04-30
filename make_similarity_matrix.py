import pandas as pd
import similarity_measures

def main():
    df_main_ta = pd.read_csv("textanalysis_20.csv")
    s_ta = df_main_ta['tokenized_sentence'].dropna()

    df_main_tr = pd.read_csv("textretrieval_20.csv")
    s_tr = df_main_tr['tokenized_sentence'].dropna()

    s = list(s_ta) + list(s_tr)
    print len(s), len(s)*len(s)

    test_sentences = s
    tfidf_mat = similarity_measures.tfidf(test_sentences)
    print "DONE"
    sem_mat = similarity_measures.semantic_similarity(test_sentences)


    sim_mat = 0.5 * (tfidf_mat + sem_mat)
    sim_mat.dump("similarity.dat")

if __name__ == '__main__':
    main()

