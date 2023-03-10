import re
import nltk
import pandas as pd

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and
    ##characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    # back to string from list
    text = " ".join(lst_text)
    return text


def oneHotEncoding(model, expert_labels_column, pred_column_name):
    """
    Question: why use this method to evaluate the model under multi-label setting?
              Why not just compare the two matrices of (NumExamples, NumLabels)?
    """
    converted_y_true = []
    converted_y_scores = []
    df = pd.read_csv(model, encoding='unicode_escape')
    for _, row in df.iterrows():
        y_true = []
        y_scores = []
        # all possible labels
        _classes = ['ankle', 'back', 'eye', 'hand', 'knee', 'other', 'shoulder']
        # for each label
        for bodyPart in _classes:
            # test if the expert annotate the example as this label, if it is then 1, else 0. Why?
            if str(row[expert_labels_column]).lower().find(bodyPart) != -1:
                y_true.append(1)
            if str(row[expert_labels_column]).lower().find(bodyPart) == -1:
                y_true.append(0)
            if str(row[pred_column_name]).lower().find(bodyPart) != -1:
                y_scores.append(1)
            if str(row[pred_column_name]).lower().find(bodyPart) == -1:
                y_scores.append(0)
        converted_y_true.append(y_true)
        converted_y_scores.append(y_scores)
    return converted_y_true, converted_y_scores
