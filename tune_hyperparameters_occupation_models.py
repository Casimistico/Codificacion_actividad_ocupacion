"""Finds the best hyperparameters for the training model for occupation code"""


import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SpanishStemmer




def get_stop_words():
    """Gets stop words to a list"""
    stop_words = []
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, ".\data\stop_words_spanish.txt")
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())
    return stop_words


def tokenizer_steam_words(words):
    """Tokenizes and steams words from text"""
    tokenizer = RegexpTokenizer(r"\w+")
    words = tokenizer.tokenize(words)
    stop = get_stop_words()
    if len(words) > 1:
        tokens = [x for x in words if x.lower() not in stop]
    else:
        tokens = words
    stemer = SpanishStemmer()
    steamed = []
    for token in tokens:
        steamed.append(stemer.stem(token))
    return " ".join(steamed)


def preprocess_text(df, column):
    """Preprocess the data"""
    df[column] = df[column].str.strip()
    df[column] = df[column].apply(tokenizer_steam_words)
    return df


def parameter_search(df, text, label, result, index):
    """This function searchs for the best parameters to predict
    a clasification code from a dataframe with text. Text is the column name
    of the text to be analyzed and label is the column name of the labels"""
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", SGDClassifier()),  ##tol=1e-3
        ]
    )

    ## This are the parameters to evaluate, the more the parameters, more time spend -----<
    parameters = {
        "vect__max_df": (0.5, 0.75, 1.0),
        "vect__max_features": (None, 5000, 10000, 50000),
        "vect__ngram_range": ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
        "tfidf__use_idf": (True, False),
        "tfidf__norm": ("l1", "l2"),
        "clf__alpha": (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001),
        "clf__penalty": ("l1", "l2", "elasticnet"),
        "clf__max_iter": (10, 50, 100),
    }

    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)

    grid_search.fit(df[text], df[label])

    int_result = pd.DataFrame()

    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        if param_name == "vect__ngram_range":
            int_result.at[index, param_name] = best_parameters[param_name][1]
        else:
            int_result.at[index, param_name] = best_parameters[param_name]

    result = pd.concat([result, int_result])

    return result


def parameter_eval_third_layer(df, df_eval):
    """Find the best hyperparameters for each case of the third layer"""
    for first in range(10):
        for second in range(10):
            for third in range(10):
                select = select_third_layer_data(df, first, second, third)
                if len(select["Cuarta"].unique()) > 1:
                    index = str(first) + str(second) + str(third)
                    df_eval = parameter_search(
                        select, "texto", "Cuarta", df_eval, index
                    )

    return df_eval


def parameter_eval_second_layer(df, df_eval):
    """Find the best hyperparameters for each case of the second layer"""
    for first in range(10):
        for second in range(10):
            select = select_second_layer_data(df, first, second)
            if len(select["Tercera"].unique()) > 1:
                index = str(first) + str(second)
                # df_eval.at[index,"indice"] = index
                df_eval = parameter_search(select, "texto", "Tercera", df_eval, index)
    return df_eval


def parameter_eval_first_layer(df, df_eval):
    """Find the best hyperparameters for each value of the first layer"""
    for first in range(10):
        select = select_first_layer_data(df, first)
        if len(select["Segunda"].unique()) > 1:
            index = str(first)
            df_eval = parameter_search(select, "texto", "Segunda", df_eval, index)
    return df_eval


def parameter_eval_base_layer(df, df_eval):
    """Find the best hyperparameters for the base layer"""
    index = "Base"
    print("Comienzando por la capa base")
    df_eval = parameter_search(df, "texto", "Primera", df_eval, index)
    print("Buscado de los parametros ")
    return df_eval


def load_data(path_to_data=".\data\Training ocupacion dataset.xlsx"):
    """Loads the data for hypertuning"""
    df = pd.read_excel(path_to_data, engine="openpyxl")
    df = preprocess_text(df, "TEXTO_OCUPACION")
    df["F72_2_N"] = df["CODIGO ACTIVIDAD"] / 9900  # Max de codigo actividad
    df["texto"] = df["TEXTO_OCUPACION"]

    df.drop(columns=["TEXTO_OCUPACION", "CODIGO ACTIVIDAD"], inplace=True)
    return df


def parameter_search_all_ocupation_models():
    """Finds the best hyperparameters for the models"""
    df = load_data()
    df_eval = pd.DataFrame()
    df_eval = parameter_eval_base_layer(df, df_eval)
    df_eval = parameter_eval_first_layer(df, df_eval)
    df_eval = parameter_eval_second_layer(df, df_eval)
    df_eval = parameter_eval_third_layer(df, df_eval)
    path = ".\\modelos_ocupacion\\parametros.xlsx"
    # df_eval.to_excel(path)


def select_first_layer_data(df, primera):
    """Selects the first layer of the data which match the input"""
    result = df[(df["Primera"] == primera)].copy()
    return result


def select_second_layer_data(df, primera, segunda):
    """Selects the first and second layer of the data which match the input"""
    result = df[(df["Primera"] == primera) & (df["Segunda"] == segunda)].copy()
    return result


def select_third_layer_data(df, primera, segunda, tercera):
    """Selects the first, second and third layer of the data which match the input"""
    result = df[
        (df["Primera"] == primera)
        & (df["Segunda"] == segunda)
        & (df["Tercera"] == tercera)
    ].copy()
    return result


if __name__ == "__main__":
    parameter_search_all_ocupation_models()
