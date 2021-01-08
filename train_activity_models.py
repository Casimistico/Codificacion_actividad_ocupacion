"""This model creates models for branch of activity predictions"""


from glob import glob
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SpanishStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, f1_score

from imblearn.over_sampling import SMOTE


def get_stop_words():
    """Loads stop words data"""
    stop_words = []
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, ".\data\stop_words_spanish.txt")
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())
    return stop_words


def tokenizer_steam_words(words):
    """Tokenize and steam texts"""
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
    """Tokenize and steam texts of dataframe"""
    df[column] = df[column].str.strip()
    df[column] = df[column].apply(tokenizer_steam_words)
    return df


def load_data():
    """Loads an process data from the training set"""
    train_data_path = ".\data\Training actividad dataset.xlsx"
    dirname = os.path.dirname(__file__)
    train_data_path = os.path.join(dirname, train_data_path)
    df = pd.read_excel(train_data_path, engine="openpyxl")
    df = preprocess_text(df, "TEXTO_ACTIVIDAD")
    return df


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


def train_best_model(df_train, target_layer, column_text, index):
    """Traint  models using the hyperparameters from a file"""
    parameters = pd.read_excel(
        ".\modelos_actividad\\parametros.xlsx", engine="openpyxl"
    )
    parameters.set_index("Unnamed: 0", inplace=True)
    clf__alpha = parameters.loc[index]["clf__alpha"]
    clf__penalty = parameters.loc[index]["clf__penalty"]
    tfidf__norm = parameters.loc[index]["tfidf__norm"]  #
    tfidf__use_idf = parameters.loc[index]["tfidf__use_idf"]  #
    vect__max_df = parameters.loc[index]["vect__max_df"]  #
    vect__max_features = parameters.loc[index]["vect__max_features"]  #
    vect__ngram_range = parameters.loc[index]["vect__ngram_range"]  #

    try: 
        vect__max_features = int(vect__max_features) #Catches the exception when this value is Nan
    except:
        None 

    if np.isnan(tfidf__use_idf):
        tfidf__use_idf = None
    if np.isnan(vect__max_features):
        vect__max_features = None

    X = df_train[column_text]
    Y = df_train[target_layer]

    cat_val = 0

    while cat_val <= 1:

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=np.random.randint(100)
        )
        cat_val = len(y_train.unique())
        # print("Categorias {}".format(cat_val))

    ngram_range = (1, vect__ngram_range)
    print("Cargados los parametros")
    vect = CountVectorizer(
        max_df=vect__max_df, ngram_range=ngram_range, max_features=vect__max_features
    ).fit(X_train)

    X_train_vectorized = vect.transform(X_train)
    print("Generado el vector")

    tfidf_transformer = TfidfTransformer(use_idf=tfidf__use_idf, norm=tfidf__norm)

    tfidf_transformer = tfidf_transformer.fit(X_train_vectorized)
    X_train_vectorized = tfidf_transformer.transform(X_train_vectorized)
    print("Comienza la entrenamiento")
    model = SGDClassifier(
        alpha=clf__alpha, max_iter=150, penalty=clf__penalty, tol=1e-3
    )
    model.fit(X_train_vectorized, y_train)

    X_test_vectorized = vect.transform(X_test)

    print("Comienza la evaluación")
    acc, f1 = eval_model(model, X_test_vectorized, y_test, X_test)
    print("Model trained,accurracy: {:.3f}".format(acc))
    return model, vect, acc, f1


def train_model_smote(df_train, target_layer, column_text):
    """ Trains the model usen SMOTE, only used in cases with little training data"""
    X = df_train[column_text]
    Y = df_train[target_layer]

    cat_val = 0

    while cat_val <= 1:

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=np.random.randint(100)
        )
        cat_val = len(y_train.unique())
        print("Categorias {}".format(cat_val))

    ### ---------------------- Duplicate data in order to SMOTE to have enought neighbours --------------

    y_train2 = y_train.copy()
    y_train3 = y_train.copy()
    y_train4 = y_train.copy()
    y_train5 = y_train.copy()
    y_train6 = y_train.copy()
    y_train7 = y_train.copy()
    y_train = pd.concat(
        [y_train, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7]
    )

    X_train2 = X_train.copy()
    X_train3 = X_train.copy()
    X_train4 = X_train.copy()
    X_train5 = X_train.copy()
    X_train6 = X_train.copy()
    X_train7 = X_train.copy()
    X_train = pd.concat(
        [X_train, X_train2, X_train3, X_train4, X_train5, X_train6, X_train7]
    )

    ### -------------------------------------------------------------------------------------

    vect = CountVectorizer(max_df=0.75, ngram_range=(1, 3)).fit(X_train)

    X_train_vectorized = vect.transform(X_train)

    smote = SMOTE()
    X_train_vectorized_smote, y_train_smote = smote.fit_sample(
        X_train_vectorized, y_train
    )

    print("Comienza la entrenamiento")

    model = SGDClassifier(alpha=1e-06, max_iter=100, penalty="elasticnet", tol=1e-3)
    model.fit(X_train_vectorized_smote, y_train_smote)

    X_test_vectorized = vect.transform(X_test)

    print("Comienza la evaluación")
    acc, f1 = eval_model(model, X_test_vectorized, y_test, X_test)
    print("Model trained,accurracy: {:.3f}".format(acc))
    return model, vect, acc, f1


def eval_model(model, X_test_vectorized_stack, y_test, x_test):
    """Evaluates a model, returns accuracy and F1 score"""
    y_pred = model.predict(X_test_vectorized_stack)
    acc = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="macro")
    return acc, f1


def save_model(model, name):
    """Saves an object"""
    name += ".pickle"
    print(name)
    save_classifier = open(name, "wb")
    pickle.dump(model, save_classifier)
    save_classifier.close()


def train_base_layer_classifier(df):
    """Trains the base layer of activity branch"""
    name_model = ".\modelos_actividad\\base_layer_model"
    name_vect = ".\modelos_actividad\\base_layer_vect"
    scores_path = ".\scores_actividad\\"
    df_eval = pd.DataFrame()
    # df.drop(columns=['Segunda','Tercera','Cuarta','TEXTO_ACTIVIDAD','CANTIDAD','CODIGO_OCUPACION'],inplace=True)
    trained_model, vect, acc, f1 = train_best_model(
        df, "Primera", "TEXTO_ACTIVIDAD", "Base"
    )
    df_eval.at["base", "acc"] = acc
    df_eval.at["base", "f1_score"] = f1
    save_model(trained_model, name_model)
    save_model(vect, name_vect)
    print("Entrenando base guardado con exito")
    df_eval.to_excel(scores_path + "base_layer_scores.xlsx", engine="openpyxl")
    print("Archivo scores guardado con exito")


def train_first_layer_classifier(df):
    """Trains the first layer of activity branch"""
    path = ".\modelos_actividad\\"
    scores_path = ".\\scores_actividad\\"
    model_base = "first_layer_{}_model"
    vect_base = "first_layer_{}_vect"
    df_eval = pd.DataFrame()
    for first in range(10):
        select = select_first_layer_data(df, first)
        if len(select) >= 1:
            print("Primera {}".format(first))
            index = "{}".format(first)
            model_name = path + model_base.format(first)
            vect_name = path + vect_base.format(first)
            trained_model, vect, acc, f1 = train_best_model(
                select, "Segunda", "TEXTO_ACTIVIDAD", index
            )
            save_model(trained_model, model_name)
            save_model(vect, vect_name)
            df_eval.at[first, "acc"] = acc
            df_eval.at[first, "f1_score"] = f1
            print("Entrenando {} guardado con exito".format(model_name))
    df_eval.to_excel(scores_path + "first_layer_scores.xlsx", engine="openpyxl")
    print("Archivo scores guardado con exito")


def train_second_layer_classifier(df):
    """Trains the second layer of activity branch"""
    path = ".\modelos_actividad\\"
    scores_path = ".\scores_actividad\\"
    model_base = "first_layer_{}_second_layer_{}_model"
    vect_base = "first_layer_{}_second_layer_{}_vect"
    df_eval = pd.DataFrame()
    for first in range(10):
        for second in range(10):
            select = select_second_layer_data(df, first, second)
            if len(select["Tercera"].unique()) > 1:
                print("Primera {}, segunda {}".format(first, second))
                index = str(first) + str(second)
                model_name = path + model_base.format(first, second)
                vect_name = path + vect_base.format(first, second)
                (
                    trained_model_smote,
                    vect_smote,
                    acc_smote,
                    f1_smote,
                ) = train_model_smote(select, "Tercera", "TEXTO_ACTIVIDAD")
                (
                    trained_model_simple,
                    vect_simple,
                    acc_simple,
                    f1_simple,
                ) = train_best_model(select, "Tercera", "TEXTO_ACTIVIDAD", index)
                if acc_smote > acc_simple:
                    trained_model = trained_model_smote
                    vect = vect_smote
                    acc = acc_smote
                    f1 = f1_smote
                    print("Se usa SMOTE")
                else:
                    trained_model = trained_model_simple
                    vect = vect_simple
                    acc = acc_simple
                    f1 = f1_simple
                save_model(trained_model, model_name)
                save_model(vect, vect_name)
                df_eval.at[index, "acc"] = acc
                df_eval.at[index, "f1_score"] = f1
                print("Entrenando {} guardado con exito".format(model_name))

    df_eval.to_excel(scores_path + "second_layer_scores.xlsx", engine="openpyxl")
    print("Archivo scores guardado con exito")


def train_third_layer_classifier(df):
    """Trains the third layer of activity branch"""
    path = ".\modelos_actividad\\"
    scores_path = ".\scores_actividad\\"
    model_base = "first_layer_{}_second_layer_{}_third_layer_{}_model"
    vect_base = "first_layer_{}_second_layer_{}_third_layer_{}_vect"
    df_eval = pd.DataFrame()
    for first in range(10):
        for second in range(10):
            for third in range(10):
                select = select_third_layer_data(df, first, second, third)
                if len(select["Cuarta"].unique()) > 1:
                    print(
                        "Primera {}, segunda {}, tercera {}".format(
                            first, second, third
                        )
                    )
                    index = str(first) + str(second) + str(third)
                    model_name = path + model_base.format(first, second, third)
                    vect_name = path + vect_base.format(first, second, third)
                    (
                        trained_model_smote,
                        vect_smote,
                        acc_smote,
                        f1_smote,
                    ) = train_model_smote(select, "Cuarta", "TEXTO_ACTIVIDAD")
                    (
                        trained_model_simple,
                        vect_simple,
                        acc_simple,
                        f1_simple,
                    ) = train_best_model(select, "Cuarta", "TEXTO_ACTIVIDAD", index)
                    if acc_smote > acc_simple:
                        print(
                            "Se usa SMOTE, {:.3f} simple, {:.3f} SMOTE".format(
                                acc_simple, acc_smote
                            )
                        )
                        trained_model = trained_model_smote
                        vect = vect_smote
                        acc = acc_smote
                        f1 = f1_smote
                    else:
                        trained_model = trained_model_simple
                        vect = vect_simple
                        acc = acc_simple
                        f1 = f1_simple
                    save_model(trained_model, model_name)
                    save_model(vect, vect_name)
                    df_eval.at[index, "acc"] = acc
                    df_eval.at[index, "f1_score"] = f1
                    print("Entrenando {} guardado con exito".format(model_name))
    df_eval.to_excel(scores_path + "third_layer_scores.xlsx", engine="openpyxl")
    print("Archivo scores guardado con exito")


def train_activity_models():
    """Trains models than predict a code of activity branch from a context"""
    models_folder = ".\modelos_actividad\\"
    all_files = glob(models_folder + "/*.pickle")
    print("Limpiando carpeta de modelos de actividad")
    for file in all_files:
        os.remove(file)
    print("Cargando datos")
    df = load_data()
    print("Datos cargados comienza el entrenamiento")
    train_base_layer_classifier(df)
    train_first_layer_classifier(df)
    train_second_layer_classifier(df)
    train_third_layer_classifier(df)


if __name__ == "__main__":
    train_activity_models()
