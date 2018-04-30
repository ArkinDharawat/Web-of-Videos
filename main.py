import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template


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

if __name__ == "__main__":
    app.run()
