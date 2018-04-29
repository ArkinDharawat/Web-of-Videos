import pandas as pd
import sys

def total_time(row):
    """Calculate the total seconds taken by the text segment

    Args:
        row (pandas.Series): the row of the Dataframe as a Series

    Returns:
        int: the total time occupied by the text segment
    """
    start_time_list =  map(int, row['start_time'].split(":"))
    end_time_list = map(int, row['end_time'].split(":"))


    # break up the times and subtract them
    return (end_time_list[2] + end_time_list[1]*60 + end_time_list[0]*3600) - \
           (start_time_list[2] + start_time_list[1]*60 + start_time_list[0]*3600)


# TODO: Optimize using reduce?

def make_row(time_accumalted, token_sentence_accumalted, sentence_accumalted, final_start_time,
             final_end_time, lecture):
    """Make the row for the dataframe

    Args:
        time_accumalted (int):  the total time
        token_sentence_accumalted (str): the tokenized sentence
        sentence_accumalted (str): the sentence
        final_start_time (str): the final start time
        final_end_time (str): the final end time
        lecture (str):  the lecture name

    Returns:
        Dictionarty: the dictionary with keys as columns and values as parameters
    """
    new_row = {}
    new_row['total_time_taken'] = time_accumalted
    # print token_sentence_accumalted
    new_row['tokenized_sentence'] = repr(token_sentence_accumalted[1:])
    new_row['sentence'] = sentence_accumalted[1:]
    new_row['start_time'] = final_start_time[0]
    new_row['end_time'] = final_end_time[-1]
    new_row['lecture'] = lecture

    return new_row


def reduce_dataframe(df_main, time_gap):
    """Coalesce rows of the Dataframe that are below the time_gap

    Args
        df_main (pandas.Dataframe): the Dataframe that we want to reduce
        time_gap (int): the minimum time occupied by the reduced rows

    Return
        pandas.Dataframe: the reduced Dataframe
    """

    df_main['total_lecture_time'] = df_main.apply(total_time, axis=1)

    df_new = pd.DataFrame()

    # reduce the dataframe by lectures
    lectures = set(df_main['lecture_name'])



    for lecture in lectures:

        df_subset = df_main[df_main['lecture_name'] == lecture]

        time_accumalted = 0
        sentence_accumalted = ""
        token_sentence_accumalted = ""

        final_end_time = [df_main.iloc[df_subset.index[0]]['end_time']]
        final_start_time = [df_main.iloc[df_subset.index[0]]['start_time']]

        # print df_subset['total_lecture_time']


        for i in df_subset.index[1:]:
             row = df_main.iloc[i]

             if (time_accumalted < time_gap):
                 time_accumalted += row['total_lecture_time']
                 sentence_accumalted = sentence_accumalted + " " + row['sentence']

                 try:
                     token_sentence_accumalted = token_sentence_accumalted + " " + row['tokenized_sentence']
                 except:
                     pass

                 final_end_time.append(row['end_time'])
                 final_start_time.append(row['start_time'])
             else:
                new_row = make_row(time_accumalted, token_sentence_accumalted, sentence_accumalted, final_start_time,
                                   final_end_time, lecture)


                df_new = df_new.append(new_row, ignore_index=True)


                time_accumalted = 0
                sentence_accumalted = ""
                token_sentence_accumalted = ""
                final_end_time = [row['end_time']]

                final_start_time = [row['start_time']]

        new_row = make_row(time_accumalted, token_sentence_accumalted, sentence_accumalted, final_start_time,
                           final_end_time, lecture)

        df_new = df_new.append(new_row, ignore_index=True)

        # break

    return df_new

def main(df_path, time_gap):

    df_reduced = reduce_dataframe(pd.read_csv(df_path), int(time_gap))

    df_reduced.to_csv(df_path.split(".")[0] + "_" + str(time_gap) + ".csv", index=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

