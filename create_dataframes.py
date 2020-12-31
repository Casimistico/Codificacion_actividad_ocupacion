import os
import pickle
import pandas as pd

def create_ocupation_training_dataset(train):
    """Creates dataframe for training models for ocupation"""
    ocupa = train.copy()
    ocupa['Primera'] = train['F71_2'].str[:1]
    ocupa['Segunda'] = train['F71_2'].str[1:2]
    ocupa['Tercera'] = train['F71_2'].str[2:3]
    ocupa['Cuarta'] = train['F71_2'].str[3:4]
    ocupa.drop(columns=['F72_1'],inplace=True)
    ocupa.rename({'F71_1':'TEXTO_OCUPACION',"F72_2":"CODIGO ACTIVIDAD"},axis=1,inplace=True)
    ocupa.to_excel('.\data\Training ocupacion dataset.xlsx',index=False)
    print('Archivo training ocupacion guardado con exito')
     
    create_vocab(ocupa,'TEXTO_OCUPACION',"Ocupacion") 
    print("Creado el vocabulario de ocupacion")


def create_activation_training_dataset(train):
    """Creates dataframe for training models for activity"""
    acti = train.copy()
    acti['Primera'] = train['F72_2'].str[:1]
    acti['Segunda'] = train['F72_2'].str[1:2]
    acti['Tercera'] = train['F72_2'].str[2:3]
    acti['Cuarta'] = train['F72_2'].str[3:4]
    acti.rename({'F72_1':'TEXTO_ACTIVIDAD',"F72_2":"CODIGO ACTIVIDAD",},axis=1,inplace=True)
    acti.drop(columns=['F71_1'],inplace=True)
    acti.dropna(axis=0,inplace=True)
    
    acti.to_excel('.\data\Training actividad dataset.xlsx',index=False)
    print('Archivo training actividad guardado con exito')
    
    create_vocab(acti,'TEXTO_ACTIVIDAD',"Actividad")
    print("Creado el vocabulario de actividad")


def create_vocab(df,text_column,name):
    """Creates vocabulary file"""
    
    vocab = []
    for text in df[text_column].unique():
        for word in text.split():
            if word not in vocab:
                vocab.append(word)
    name = '.\data\Vocabulario ' + name + '.txt'
    with open(name, 'w') as f:
        for item in sorted(vocab):
            f.write("%s\n" % item)
    f.close()

def get_relative_path(relative_path):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, relative_path)


def create_dataframes(path='.\data\Datos depurados a mano.xlsx',val_size=10000):

    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '.\data\Datos depurados a mano.xlsx')
    
    df = pd.read_excel(path,dtype={'F71_2':str,'F72_2':str},engine='openpyxl')
    df.dropna(axis=0,inplace=True)
    print("Cargado datos")
    print("Registros totales en archivo: {} ".format(df.shape[0]))
    
    df = df.sample(frac=1).reset_index(drop=True) #Shuffle dataset
    df["Combinaciones"] =  df['F71_2'] + df['F72_2']
    
    os.remove('.\data\Combinaciones permitidas.txt')
    with open('.\data\Combinaciones permitidas.txt', 'w') as f:
        for item in sorted(df["Combinaciones"].unique()):
            f.write("%s\n" % item)
            
    print("Combinaciones guardadas")
    
    n_train = df.shape[0] - val_size
    train = df.loc[:n_train]
    val = df.loc[n_train:]
    path_validation_data = '.\data\Datos para validacion.xlsx'
    val.to_excel(path_validation_data,index=False)
    create_ocupation_training_dataset(train)
    create_activation_training_dataset(train)
    print('Creado datasets con exito')


if __name__ == "__main__":
    original_file = ".\data\Todo el dataset sin nan.xlsx"
    mod_file = '.\data\Datos depurados a mano.xlsx'
    create_dataframes(mod_file)


