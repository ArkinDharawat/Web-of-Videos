import pandas as pd
import numpy as np

df_textanalysis = pd.read_csv("textanalysis_20.csv")
df_textretrieval = pd.read_csv("textretrieval_20.csv")
similarity_mat = np.load("similarity.dat")
Y = 0.5

sentences = list(df_textanalysis['sentence']) + list(df_textretrieval['sentence'])
print len(sentences)

def search_topic(idx):
    if idx > df_textanalysis.shape[0]:
        return df_textretrieval.iloc[idx-df_textanalysis.shape[0]]['lecture']
    else:
        return df_textanalysis.iloc[idx]['lecture']

def write_to_file():
    file_sim_results = open("sim_res.txt", "w")

    for i in range(len(similarity_mat)):
        sim_list = similarity_mat[i]
        indices = []
        for j in range(len(sim_list)):
            if sim_list[j] > Y and sim_list[j] < 1:
                indices.append(j)
        if indices!=[]:
            node_topic = search_topic(i)
            file_sim_results.write("Node: index {0} and lecture {1}\n".format(i, search_topic(i)))
            for idx in indices:
                if search_topic(idx)!=node_topic:
                    file_sim_results.write("Neighbour: index {0} and lecture {1}\n".format(idx, search_topic(idx)))
        file_sim_results.write("\n")
        # break
    file_sim_results.close()


if __name__ == '__main__':
    write_to_file()
