import sys, os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


sentence_tokenizer = RegexpTokenizer(r'\w+')
filename_tokenizer = RegexpTokenizer(r"(\d+) \- (\d+) \- (\d+\.\d+) (.*) |(\d+) \- (\d+) \- (.*) ") #TODO: Simplify this expression???

stop_words = stopwords.words('english')

column_names = ["lecture", "sub_lecture", "lecture_no", "lecture_name", 'start_time', 'end_time', 'sentence', 'tokenized_sentence']

def remove_stop_words(word):
    """Remove stop words and punctuations from a sentence

    Args:
        word (str): the sentence to be formatted
    Returns:
        str: the formatted string
    """
    word_list = sentence_tokenizer.tokenize(word)
    filtered_words = [word for word in word_list if word not in stop_words]
    return ' '.join(filtered_words)

def format_node(node):
    """format the lines that have been read from the file

       Args:
           node (list): a list of lines read
       Returns:
           list: list with start time, end time, sentence extracted
       """
    time_range = node[0].split(" --> ")
    start_time = time_range[0].split(",")[0]
    end_time = time_range[1].split(",")[0]

    line_str = ' '.join(node[1:])

    return [start_time, end_time, line_str, remove_stop_words(line_str)]

# text retrival
def main(arg):
    transcript_folder = arg
    transcript_files = os.listdir(os.path.join(os.getcwd(), transcript_folder))
    df_main = pd.DataFrame()

    for file in transcript_files:
        print file
        toeknized_filename = filename_tokenizer.tokenize(file)[0]
        if toeknized_filename[0]!='':
            lecture, sub_lecture, lecture_no, lecture_name = toeknized_filename[0:4]
        else:
            lecture, sub_lecture, lecture_name = toeknized_filename[4:7]
            lecture_no = ''

        with open(os.path.join(transcript_folder, file)) as f_in:
            lines = filter(None, (line.rstrip() for line in f_in))

        format_list = []
        nodes = []
        for x in lines:
            if x.isdigit():
                if format_list!=[]:
                    formatted_data = format_node(format_list)
                    nodes.append(formatted_data)
                    format_list = []
            else:
                format_list.append(x)

        for x in nodes:
            df_row = {}
            row_lst = [lecture, sub_lecture, lecture_no, lecture_name] + x
            for i in range(len(row_lst)):
                df_row[column_names[i]] = row_lst[i]

            df_main = df_main.append(df_row, ignore_index=True)

    df_main.to_csv(transcript_folder[:-4]+".csv")

if __name__ == '__main__':
    main(sys.argv[1])