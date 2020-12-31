import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import pickle
from scipy import sparse
import ast
from glob import glob
import os



def get_stop_words():
    """Loads stop words data"""
    stop_words = []
    with open('.\data\stop_words_spanish.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())
    return stop_words

def tokenizer_steam_words(words):
    """Tokenize and steam texts"""
    tokenizer = RegexpTokenizer(r'\w+')
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
    return ' '.join(steamed)


def preprocess_text(df,column):
    """Tokenize and steam texts of dataframe"""
    df[column] = df[column].str.strip()
    df[column] =  df[column].apply(tokenizer_steam_words)
    return df

def load_data():
    train_data_path = '.\data\Training ocupacion dataset.xlsx'
    dirname = os.path.dirname(__file__)
    train_data_path = os.path.join(dirname, train_data_path)
    df = pd.read_excel(train_data_path,engine='openpyxl')
    df = preprocess_text(df,'TEXTO_OCUPACION')
    df['F72_2_N'] = df['CODIGO ACTIVIDAD']/  9900 #Max de codigo actividad
    df['texto'] = df['TEXTO_OCUPACION']
    df.drop(columns=['TEXTO_OCUPACION',"CODIGO ACTIVIDAD"],inplace=True)
    return df

def train_best_model(df_train,target_layer,column_text,index):
    
    parameters = pd.read_excel(".\modelos_ocupacion\parametros.xlsx",engine='openpyxl')
    parameters.set_index('Unnamed: 0',inplace=True)
    clf__alpha = parameters.loc[index]['clf__alpha']
    clf__penalty = parameters.loc[index]['clf__penalty'] 
    tfidf__norm = parameters.loc[index]['tfidf__norm'] #
    tfidf__use_idf =  parameters.loc[index]['tfidf__use_idf'] #
    vect__max_df = parameters.loc[index]['vect__max_df'] #
    vect__max_features = parameters.loc[index]['vect__max_features'] #
    vect__ngram_range = parameters.loc[index]['vect__ngram_range'] #
    
    try:
        vect__max_features = int(vect__max_features)
    except:
        None
    if np.isnan(tfidf__use_idf ):
        tfidf__use_idf  = None
    if np.isnan(vect__max_features):
        vect__max_features = None
    
    X = df_train[[column_text,"F73",'F72_2_N']]
    Y = df_train[target_layer]
    
    cat_val = 0
    
    while cat_val <=1 :
    
        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=np.random.randint(100))
        cat_val = len(y_train.unique())
        #print("Categorias {}".format(cat_val))
    
    
    ngram_range = (1, vect__ngram_range)
    print("Cargados los parametros")
    vect = CountVectorizer(max_df=vect__max_df, ngram_range=ngram_range,max_features=vect__max_features).fit(X_train[column_text])
    
    X_train_vectorized = vect.transform(X_train[column_text])
    print('Generado el vector')
    
    tfidf_transformer = TfidfTransformer(use_idf=tfidf__use_idf,norm = tfidf__norm) 
        
    tfidf_transformer = tfidf_transformer.fit(X_train_vectorized)
    X_train_vectorized = tfidf_transformer.transform(X_train_vectorized)
    
    X_train_vectorized_stack = sparse.hstack((X_train_vectorized,sparse.coo_matrix(X_train["F73"]).reshape(X_train.shape[0],1)))
    X_train_vectorized_stack = sparse.hstack((X_train_vectorized_stack,sparse.coo_matrix(X_train['F72_2_N']).reshape(X_train.shape[0],1)))
    
    
    print("Comienza la entrenamiento")
    model = SGDClassifier(alpha=clf__alpha, max_iter=300, penalty=clf__penalty, tol=1e-6)
    model.fit(X_train_vectorized_stack, y_train)
    
    del X_train_vectorized_stack
    print("Limpieza de memoria")
    
    X_test_vectorized = vect.transform(X_test[column_text])
    X_test_vectorized = tfidf_transformer.transform(X_test_vectorized)
    
    X_test_vectorized_stack = sparse.hstack((X_test_vectorized,sparse.coo_matrix(X_test["F73"]).reshape(X_test.shape[0],1)))
    X_test_vectorized_stack = sparse.hstack((X_test_vectorized_stack,sparse.coo_matrix(X_test['F72_2_N']).reshape(X_test.shape[0],1)))
 

    
    print("Comienza la evaluación")
    acc, f1 = eval_model(model,X_test_vectorized_stack,y_test, X_test)
    print("Model trained,accurracy: {:.3f}".format(acc))
    return model, vect, acc, f1

def select_first_layer_data(df,primera):
    result = df[(df['Primera']==primera)].copy()
    return result

def select_second_layer_data(df,primera,segunda):
    result = df[(df['Primera']==primera)&(df['Segunda']==segunda)].copy()
    return result

def select_third_layer_data(df,primera,segunda,tercera):
    result = df[(df['Primera']==primera)&(df['Segunda']==segunda)&(df['Tercera']==tercera)].copy()
    return result

def eval_model(model,X_test_vectorized_stack,y_test,x_test):
    y_pred = model.predict(X_test_vectorized_stack)
    acc = accuracy_score(y_pred,y_test)
    f1 = f1_score(y_pred,y_test,average='macro')
    return acc, f1

def save_model(model,name):
    name += ".pickle"
    save_classifier = open(name, "wb")
    pickle.dump(model, save_classifier)
    save_classifier.close()


def train_base_layer_classifier(df):
    name_model = ".\modelos_ocupacion\\base_layer_model"
    name_vect = ".\modelos_ocupacion\\base_layer_vect"
    scores_path = ".\scores_ocupacion\\"
    df_eval = pd.DataFrame()
    trained_model, vect, acc, f1 = train_best_model(df,'Primera','texto','Base')
    df_eval.at['base','acc'] = acc
    df_eval.at['base','f1_score'] = f1
    save_model(trained_model,name_model)
    save_model(vect,name_vect)
    print("Entrenando base guardado con exito")
    df_eval.to_excel(scores_path + 'base_layer_scores.xlsx',engine='openpyxl')
    print('Archivo scores guardado con exito')

def train_first_layer_classifier(df):
    path = ".\modelos_ocupacion\\"
    scores_path = ".\scores_ocupacion\\"
    model_base = "first_layer_{}_model"
    vect_base = "first_layer_{}_vect"
    df_eval = pd.DataFrame()
    for first in range(10):
        select = select_first_layer_data(df,first)
        if len(select)>=1:
            model_name = path + model_base.format(first)
            vect_name = path + vect_base.format(first)
            index = str(first)
            trained_model, vect, acc, f1 = train_best_model(select,'Segunda','texto',index)
            save_model(trained_model,model_name)
            save_model(vect,vect_name)
            df_eval.at[first,'acc'] = acc
            df_eval.at[first,'f1_score'] = f1
            print("Entrenando {} guardado con exito".format(model_name))
    df_eval.to_excel(scores_path + 'first_layer_scores.xlsx',engine='openpyxl')
    print('Archivo scores guardado con exito')

def train_second_layer_classifier(df):
    path = ".\modelos_ocupacion\\"
    scores_path = ".\scores_ocupacion\\"
    model_base = "first_layer_{}_second_layer_{}_model"
    vect_base = "first_layer_{}_second_layer_{}_vect"
    df_eval = pd.DataFrame()
    for first in range(10):
        for second in range(10):
            select = select_second_layer_data(df,first,second)
            if len(select['Tercera'].unique())>1:
                index = str(first) + str(second)
                model_name = path + model_base.format(first,second)
                vect_name = path + vect_base.format(first,second)
                trained_model, vect, acc, f1 = train_best_model(select,'Tercera','texto',index)
                save_model(trained_model,model_name)
                save_model(vect,vect_name)
                df_eval.at[index,'acc'] = acc
                df_eval.at[index,'f1_score'] = f1
                print("Entrenando {} guardado con exito".format(model_name))
    df_eval.to_excel(scores_path + 'second_layer_scores.xlsx',engine='openpyxl')
    print('Archivo scores guardado con exito')

def train_third_layer_classifier(df):
    path = ".\modelos_ocupacion\\"
    scores_path = ".\scores_ocupacion\\"
    model_base = "first_layer_{}_second_layer_{}_third_layer_{}_model"
    vect_base = "first_layer_{}_second_layer_{}_third_layer_{}_vect"
    df_eval = pd.DataFrame()
    for first in range(10):
        for second in range(10):
            for third in range(10):
                select = select_third_layer_data(df,first,second,third)
                if len(select['Cuarta'].unique())>1:
                    index = str(first) + str(second) + str(third)
                    model_name = path + model_base.format(first,second,third)
                    vect_name = path + vect_base.format(first,second,third)
                    trained_model, vect, acc, f1 = train_best_model(select,'Cuarta','texto',index)
                    save_model(trained_model,model_name)
                    save_model(vect,vect_name)
                    df_eval.at[index,'acc'] = acc
                    df_eval.at[index,'f1_score'] = f1
                    print("Entrenando {} guardado con exito".format(model_name))
    df_eval.to_excel(scores_path + 'third_layer_scores.xlsx',engine='openpyxl')
    print('Archivo scores guardado con exito')

def train_ocupation_models():
    models_folder = ".\modelos_ocupacion\\"
    all_files = glob(models_folder + "/*.pickle")
    print('Limpiando carpeta de modelos de ocupación')
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
    train_ocupation_models()
