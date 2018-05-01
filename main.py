import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template
import sys


lecture =[]
start_time = []
end_time = []
sentences = []
dic = {}
app = Flask(__name__)
ALL_LECTURES = 0
SUBSET_INDICES = [0, 7, 1907, 1876, 224, 230]

@app.route('/')
def main():
    global lecture
    global start_time
    global end_time
    global sentences
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
    sentence_subset = df_main['sentence']


    indices = list(range(matrix.shape[0]))
    lecture = map(lambda x: lecture_subset[x], indices)
    start_time = map(lambda x: start_time_subset[x], indices)
    end_time = map(lambda x: end_time_subset[x], indices)
    sentences = map(lambda x: sentence_subset[x], indices)


    keys = range(matrix.shape[0])
    dic = {key : [] for key in keys}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if matrix[i][j] > 0.5 and matrix[i][j] < 1.0:
                temp = dic[i]
                temp.append(j)
                dic[i] = temp
    data = {}
    for i in range(matrix.shape[0]):
        string = "{0}-{1} Lecture: {2} // Sentence: {3}".format(start_time[i], end_time[i], lecture[i], sentences[i])
        #string = "{0}-{1} Lecture: {2}".format(start_time[i], end_time[i], lecture[i])
        data[i] = string

    if ALL_LECTURES:
        return render_template("home.html", dictionary = data)
    else:
        subset_data = {}
        for i in SUBSET_INDICES:
            subset_data[i] = data[i]

        return render_template("home.html", dictionary=subset_data)

@app.route('/<int:id>')
def search(id):
    global dic
    sim_list = dic[id]
    data = {}
    original_node = "Lecture(s) Similar to : {0}-{1} Lecture: {2} // Sentence: {3}".format(start_time[id], end_time[id],
                                                                                         lecture[id], sentences[id])
    #original_node = "Lecture(s) Similar to : {0}-{1} Lecture: {2}".format(start_time[id], end_time[id], lecture[id])

    for i in range(len(sim_list)):
        index = sim_list[i]
        string = "{0}-{1} Lecture: {2} // Sentence: {3}".format(start_time[index], end_time[index], lecture[index], sentences[index])
        data[index] = string
    return render_template("similar.html", dictionary = data, orig_node = original_node)

if __name__ == "__main__":
    if sys.argv[1] == str(1):
        ALL_LECTURES = 1
    app.run()
