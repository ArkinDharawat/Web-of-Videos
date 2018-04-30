# Web-of-Videos
Project to link scattered videos from various sources


### Overview of Functions
We have detailed function documentation in the files themselves. We used [Google Docstring Style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) to document all .py files
1. parse_text.py: 
    * main: this function parses the folder full of .srt files, extract relevant information such as lecture name, text, end time, start time etc and puts them into a .csv file. See Usage for more. 
    * format_node: format the lines that have been read from the file.
    * remove_stop_words: Remove stop words and punctuations from a sentence.
2. reduce_dataframe_maker.py: 
    Like mentioned in the project proposal, We decided to coalesce original text divisions into 20-second segments so that we have more text data to work with. This takes in a .csv file and the time of each segment. See Usage for more.
    * reduce_dataframe: Coalesce rows of the Dataframe that are below the time_gap
    * make_row: Make the row for the new dataframe
    * total_time: Calculate the total seconds taken by the text segment
3. similarity_measures.py:
    This file helps in calculating and testing the various similarity measures.
    * random_sentences: Select sentences at random from the Dataframe
    * idf: Calculate idf of given list of sentences
    * maximum_similarity: Calculate maximum similarity from word and token using Wu-Palmer metric.
    * similarity_measure: Calulcate the similarity metric between 2 terms
    * semantic_similarity: Make a matrix of semantic similarity between i and j entries
    * tfidf: Make a matrix of tfidf similarity between i and j entries 
    * w2v: Calculate similarity in Word Embeddings
    * word2vec_similarity: Calculate similarity matrix using a trained word 2 vec model
    * write_to_results: write results from the matrix generated to a file, result.txt
4. make_similarity_matrix.py:
    * main: This takes in both the sentences from both the .csvs and runs the similarity measure on them. It generates a similarity matrix and writes it to a file, similarity.dat, from which it can be loaded.
5. make_similarity_file.py:
    * main: Write the results obtained from the similarity matrix to a file, sim_res.txt We only write those results where matrix[i][j] > 0.5 and the node and it's neighbors are from different lectures.
    
### Implementation
we had the following steps in the implementation:
1. First we used the functions in our parse_text file to parse the transcript folders that Prof. Zhai had sent to us. We put each MOOC into a seperate .csv, textanalysis.csv and textretrieval.csv
2. We decided to coalesce the text segments in the csv and tested various intervals of 5, 10, 20 and 30 seconds. From all of these 20-second segments made the most sense.
3. We experimented with various similarities measure, including tf-idf, various knowledge-based similarities and Word2Vec. Word2Vec didn't perform very well, but we found [this paper](https://pdfs.semanticscholar.org/1374/617e135eaa772e52c9a2e8253f49483676d6.pdf) very helpful. After reading that we used the Wu and Palmer metric and tf-idf score to calculate similarity. 
4. We then generated a similarity matrix from the scores and the sentences and saved it to similarity.dat 
5. Website ???
### Usage
Here is how you can use or re-create what we did:
1. parse_text.py
2. reduce_dataframe_maker.py
3. similarity_measures.py
4. make_similarity_matrix.py
5. make_similarity_file.py/Website
### Member Contribution
* **abagas3**: 
* **arkinrd2**:
* **ps4**: