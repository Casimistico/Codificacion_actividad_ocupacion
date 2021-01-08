import pickle
import os
from glob import glob
from scipy import sparse
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SpanishStemmer


def get_relative_path(relative_path):
    """Gets relative path from file"""
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, relative_path)


def occupation_base_layer_prediction(text, act_code, F73_code):
    """Predicts the base layer coding for an occupation given a text and above categories"""

    model_name = get_relative_path(".\modelos_ocupacion/base_layer_model.pickle")
    vect_name = get_relative_path(".\modelos_ocupacion/base_layer_vect.pickle")
    scores = pd.read_excel(
        ".\scores_ocupacion\\base_layer_scores.xlsx", engine="openpyxl"
    )
    scores.set_index(scores.columns[0], inplace=True)
    training_score = scores.loc["base"]["acc"]
    model_f = open(model_name, "rb")
    model = pickle.load(model_f)
    model_f.close()
    vectorizer_f = open(vect_name, "rb")
    vect = pickle.load(vectorizer_f)
    vectorizer_f.close()

    # text = [text + ' ' + act_code]
    text = [text]

    vect_text = vect.transform(text)

    vect_text_stack = sparse.hstack((vect_text, F73_code))
    vect_text_stack = sparse.hstack((vect_text_stack, act_code))

    y_pred = model.predict(vect_text_stack)
    return y_pred[0], training_score


def occupation_first_layer_prediction(text, act_code, F73_code, category):
    """Predicts a first layer coding for an occupation given a text and above category"""
    models_folder = get_relative_path(".\modelos_ocupacion\\")
    all_files = glob(models_folder + "/*.pickle")
    model_name = get_relative_path(
        ".\modelos_ocupacion\\first_layer_{}_model.pickle".format(category)
    )
    vect_name = get_relative_path(
        ".\modelos_ocupacion\\first_layer_{}_vect.pickle".format(category)
    )
    if model_name in all_files:
        model_f = open(model_name, "rb")
        model = pickle.load(model_f)
        model_f.close()
        vectorizer_f = open(vect_name, "rb")
        vect = pickle.load(vectorizer_f)
        vectorizer_f.close()

        # text = [text + ' ' + act_code]
        text = [text]

        vect_text = vect.transform(text)
        vect_text_stack = sparse.hstack((vect_text, F73_code))
        vect_text_stack = sparse.hstack((vect_text_stack, act_code))

        y_pred = model.predict(vect_text_stack)
        scores = pd.read_excel(
            ".\scores_ocupacion\\first_layer_scores.xlsx", engine="openpyxl"
        )
        scores.set_index(scores.columns[0], inplace=True)
        training_score = scores.loc[category]["acc"]
        return y_pred[0], training_score
    else:
        return 1, 1


def occupation_second_layer_prediction(
    text, act_code, F73_code, first_category, second_category
):
    """Predicts a second layer coding for an occupation given a text and above categories"""
    models_folder = get_relative_path(".\modelos_ocupacion\\")
    all_files = glob(models_folder + "/*.pickle")
    model_name = get_relative_path(
        ".\modelos_ocupacion\\first_layer_{}_second_layer_{}_model.pickle".format(
            first_category, second_category
        )
    )
    vect_name = get_relative_path(
        ".\modelos_ocupacion\\first_layer_{}_second_layer_{}_vect.pickle".format(
            first_category, second_category
        )
    )
    if model_name in all_files:
        model_f = open(model_name, "rb")
        model = pickle.load(model_f)
        model_f.close()
        vectorizer_f = open(vect_name, "rb")
        vect = pickle.load(vectorizer_f)
        vectorizer_f.close()

        # text = [text + ' ' + act_code]
        text = [text]

        vect_text = vect.transform(text)
        vect_text_stack = sparse.hstack((vect_text, F73_code))
        vect_text_stack = sparse.hstack((vect_text_stack, act_code))

        y_pred = model.predict(vect_text_stack)
        scores = pd.read_excel(
            ".\scores_ocupacion\\second_layer_scores.xlsx", engine="openpyxl"
        )
        scores.set_index(scores.columns[0], inplace=True)
        index = int(str(first_category) + str(second_category))
        training_score = scores.loc[index]["acc"]
        return y_pred[0], training_score
    else:
        return 1, 1


def occupation_third_layer_prediction(
    text, act_code, F73_code, first_category, second_category, third_category
):
    """Predicts a third layer coding for an occupation given a text and above categories"""
    models_folder = get_relative_path(".\modelos_ocupacion\\")
    all_files = glob(models_folder + "/*.pickle")
    model_name = get_relative_path(
        ".\modelos_ocupacion\\first_layer_{}_second_layer_{}_third_layer_{}_model.pickle".format(
            first_category, second_category, third_category
        )
    )
    vect_name = get_relative_path(
        ".\modelos_ocupacion\\first_layer_{}_second_layer_{}_third_layer_{}_vect.pickle".format(
            first_category, second_category, third_category
        )
    )
    if model_name in all_files:
        model_f = open(model_name, "rb")
        model = pickle.load(model_f)
        model_f.close()
        vectorizer_f = open(vect_name, "rb")
        vect = pickle.load(vectorizer_f)
        vectorizer_f.close()

        # text = [text + ' ' + act_code]
        text = [text]

        vect_text = vect.transform(text)
        vect_text_stack = sparse.hstack((vect_text, F73_code))
        vect_text_stack = sparse.hstack((vect_text_stack, act_code))

        y_pred = model.predict(vect_text_stack)
        scores = pd.read_excel(
            ".\scores_ocupacion\\third_layer_scores.xlsx", engine="openpyxl"
        )
        scores.set_index(scores.columns[0], inplace=True)
        index = int(str(first_category) + str(second_category) + str(third_category))
        training_score = scores.loc[index]["acc"]
        return y_pred[0], training_score

    else:
        if "{}{}{}".format(first_category, second_category, third_category) in [
            "322"
            #"832",
        ]: 
            last = 1
        else:
            last = 0  # Todas las categorias terminan en 0 salvo la 322 ( solo en caso de eliminarse la categoria832

        return last, 1  # La categoria capa 4 tiene un 0 como valor de categoria unica


def occupation_predict_coding(text, act_code, F73_code):
    """Predicts a coding for an occupation given a text"""
    base_cat_pred, base_acc = occupation_base_layer_prediction(text, act_code, F73_code)
    first_layer_pred, first_acc = occupation_first_layer_prediction(
        text, act_code, F73_code, base_cat_pred
    )
    second_layer_pred, second_acc = occupation_second_layer_prediction(
        text, act_code, F73_code, base_cat_pred, first_layer_pred
    )
    third_layer_pred, third_acc = occupation_third_layer_prediction(
        text, act_code, F73_code, base_cat_pred, first_layer_pred, second_layer_pred
    )
    total_acc = base_acc * first_acc * second_acc * third_acc
    code = "{}{}{}{}".format(
        base_cat_pred, first_layer_pred, second_layer_pred, third_layer_pred
    )
    file = open(".\data\Diccionario CIUO 08.pickle", "rb")
    code_dict = pickle.load(file)
    file.close()
    descr = code_dict.get("Descripción").get(code)
    return code, descr, round(total_acc, 3)


def activity_base_layer_prediction(text):
    """Predicts the base layer coding for an occupation given a text and above categories"""
    model_name = get_relative_path(".\modelos_actividad\\base_layer_model.pickle")
    vect_name = get_relative_path(".\modelos_actividad\\base_layer_vect.pickle")
    scores_path = get_relative_path(".\scores_actividad\\base_layer_scores.xlsx")
    scores = pd.read_excel(scores_path, engine="openpyxl")
    scores.set_index(scores.columns[0], inplace=True)
    training_score = scores.loc["base"]["acc"]
    model_f = open(model_name, "rb")
    model = pickle.load(model_f)
    model_f.close()
    vectorizer_f = open(vect_name, "rb")
    vect = pickle.load(vectorizer_f)

    vectorizer_f.close()
    vect_text = vect.transform(text)

    y_pred = model.predict(vect_text)
    return y_pred[0], training_score


def activity_first_layer_prediction(text, first_category):
    """Predicts a first layer coding for an occupation given a text and above category"""
    models_folder = get_relative_path(".\modelos_actividad\\")
    all_files = glob(models_folder + "/*.pickle")
    model_name = get_relative_path(
        ".\modelos_actividad\\first_layer_{}_model.pickle".format(first_category)
    )
    vect_name = get_relative_path(
        ".\modelos_actividad\\first_layer_{}_vect.pickle".format(first_category)
    )
    scores_path = get_relative_path(".\scores_actividad\\first_layer_scores.xlsx")
    if model_name in all_files:
        model_f = open(model_name, "rb")
        model = pickle.load(model_f)
        model_f.close()
        vectorizer_f = open(vect_name, "rb")
        vect = pickle.load(vectorizer_f)
        vectorizer_f.close()

        vect_text = vect.transform(text)

        y_pred = model.predict(vect_text)
        scores = pd.read_excel(scores_path, engine="openpyxl")
        scores.index = scores.index.map(str)
        scores.set_index(scores.columns[0], inplace=True)
        training_score = scores.loc[first_category]["acc"]
        return y_pred[0], training_score
    else:
        return 0, 1


def activity_second_layer_prediction(text, first_category, second_category):
    """Predicts a second layer coding for an occupation given a text and above categories"""
    models_folder = get_relative_path(".\modelos_actividad\\")
    all_files = glob(models_folder + "/*.pickle")
    model_name = get_relative_path(
        ".\modelos_actividad\\first_layer_{}_second_layer_{}_model.pickle".format(
            first_category, second_category
        )
    )
    vect_name = get_relative_path(
        ".\modelos_actividad\\first_layer_{}_second_layer_{}_vect.pickle".format(
            first_category, second_category
        )
    )
    scores_path = get_relative_path(".\scores_actividad\\second_layer_scores.xlsx")
    if model_name in all_files:
        model_f = open(model_name, "rb")
        model = pickle.load(model_f)
        model_f.close()
        vectorizer_f = open(vect_name, "rb")
        vect = pickle.load(vectorizer_f)
        vectorizer_f.close()

        vect_text = vect.transform(text)

        y_pred = model.predict(vect_text)
        scores = pd.read_excel(scores_path, engine="openpyxl")
        scores.index = scores.index.map(str)
        scores.set_index(scores.columns[0], inplace=True)
        index = int(str(first_category) + str(second_category))
        training_score = scores.loc[index]["acc"]
        return y_pred[0], training_score
    else:
        return 0, 1


def activity_third_layer_prediction(
    text, first_category, second_category, third_category
):
    """Predicts a third layer coding for an occupation given a text and above categories"""
    models_folder = get_relative_path(".\modelos_actividad\\")
    all_files = glob(models_folder + "/*.pickle")
    model_name = get_relative_path(
        ".\modelos_actividad\\first_layer_{}_second_layer_{}_third_layer_{}_model.pickle".format(
            first_category, second_category, third_category
        )
    )
    vect_name = get_relative_path(
        ".\modelos_actividad\\first_layer_{}_second_layer_{}_third_layer_{}_vect.pickle".format(
            first_category, second_category, third_category
        )
    )
    scores_path = get_relative_path(".\scores_actividad\\third_layer_scores.xlsx")
    if model_name in all_files:
        model_f = open(model_name, "rb")
        model = pickle.load(model_f)
        model_f.close()
        vectorizer_f = open(vect_name, "rb")
        vect = pickle.load(vectorizer_f)
        vectorizer_f.close()

        vect_text = vect.transform(text)

        y_pred = model.predict(vect_text)
        scores = pd.read_excel(scores_path, engine="openpyxl")
        scores.set_index(scores.columns[0], inplace=True)
        index = int(str(first_category) + str(second_category) + str(third_category))
        training_score = scores.loc[index]["acc"]
        return y_pred[0], training_score

    else:

        if "{}{}{}".format(first_category, second_category, third_category) == "072":
            last = 1
        else:
            last = 0  # Todas las categorias terminan en 0 salvo la 322

        return last, 1  # La categoria capa 4 tiene un 0 como valor de categoria unica


def activate_predict_coding(text):
    """Predicts a coding for an occupation given a text"""
    assert isinstance(text, str), "El texto tiene que ser una string!"
    text = [text]
    base_cat_pred, base_acc = activity_base_layer_prediction(text)
    first_layer_pred, first_acc = activity_first_layer_prediction(text, base_cat_pred)
    second_layer_pred, second_acc = activity_second_layer_prediction(
        text, base_cat_pred, first_layer_pred
    )
    third_layer_pred, third_acc = activity_third_layer_prediction(
        text, base_cat_pred, first_layer_pred, second_layer_pred
    )
    total_acc = base_acc * first_acc * second_acc * third_acc
    code = "{}{}{}{}".format(
        base_cat_pred, first_layer_pred, second_layer_pred, third_layer_pred
    )
    file = open(".\data\Diccionario Actividad.pickle", "rb")
    code_dict = pickle.load(file)
    file.close()
    descr = code_dict.get("Descripción").get(code)
    return code, descr, round(total_acc, 3)


def tokenizer_steam_words(words):
    """tokenizes and steams words from a text"""
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


def get_stop_words():
    """Returns stop words as a list""" 
    stop_words = []
    with open(".\data\stop_words_spanish.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip().replace("'", "")
            stop_words.append(word)
    return stop_words


def get_allowed_combinations():
    """Gets the allowed combinations of occupation"""
    comb = []
    with open(".\data\Combinaciones permitidas.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip().replace("'", "")
            comb.append(word)
    return comb


def get_vocab_act():
    vocab = []
    path_vocab_activity = ".\data\Vocabulario Actividad.txt"
    with open(path_vocab_activity, "r") as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip().replace("'", "")
            vocab.append(word)
    return vocab


def get_vocab_ocu():
    """Creates the vocabulary as a list"""
    vocab = []
    path_vocab_occupation = ".\data\Vocabulario Ocupacion.txt"
    with open(path_vocab_occupation, "r") as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip().replace("'", "")
            vocab.append(word)
    return vocab


def predict_code(text_occupation, text_activity, F73_code):
    """Predicts a code from a context"""
    flag = ""

    comb_permitidas = get_allowed_combinations()

    voc_ocu = get_vocab_ocu()
    voc_act = get_vocab_act()

    not_voc_ocu = [x for x in text_occupation.split() if x not in voc_ocu]

    not_voc_act = [x for x in text_activity.split() if x not in voc_act]

    if len(not_voc_ocu) > 0:
        not_voc_ocu.insert(
            0, "No aparecen en vocabulario de entrenamiento de ocupación:"
        )
        flag += " ".join(not_voc_ocu)

    if len(not_voc_act) > 0:
        not_voc_act.insert(
            0, " No aparecen en vocabulario de entrenamiento de actividad:"
        )
        flag += " ".join(not_voc_act)

    text_occupation = tokenizer_steam_words(text_occupation)
    text_activity = tokenizer_steam_words(text_activity)

    activity_result = activate_predict_coding(text_activity)
    activity_code = int(activity_result[0]) / 9900
    occupation_result = occupation_predict_coding(
        text_occupation, activity_code, F73_code
    )


    pred_F71_2 = occupation_result[0]
    pred_F72_2 = activity_result[0]

    comb = pred_F71_2 + pred_F72_2

    if comb not in comb_permitidas:
        flag += " Genera nueva combinacion ocupación actividad"

    return occupation_result, activity_result, flag

def validate_models():
    """Starts validation process of the models"""
    print("Comienza el proceso de validación")
    path_validation_data = ".\data\Datos para validacion.xlsx"
    success_counter_all = 0
    success_counter_occupation = 0
    success_counter_activity = 0
    success_counter_occupation_3rd = 0
    success_counter_activity_3rd = 0
    success_counter_occupation_2nd = 0
    success_counter_activity_2nd = 0
    pred = pd.DataFrame()
    df = pd.read_excel(
        path_validation_data, dtype={"F72_2": str, "F71_2": str}, engine="openpyxl"
    )

    counter = 0
    for index, row in df.iterrows():
        counter += 1
        occupation_text = row["F71_1"]
        activity_text = row["F72_1"]
        F73_code = row["F73"]

        true_F71_2 = str(row["F71_2"])
        true_F72_2 = str(row["F72_2"])

        try:
            occupation_result, activity_result, flag = predict_code(
                occupation_text, activity_text, F73_code
            )
        except Exception as e:
            print(e)
            print(row["F71_1"], row["F72_1"])

        pred_F71_2 = occupation_result[0]
        pred_ocu_cat = occupation_result[1]
        pred_ocu_prob = occupation_result[2]

        pred_F72_2 = activity_result[0]
        pred_act_cat = activity_result[1]
        pred_act_prob = activity_result[2]

        if pred_F71_2 == true_F71_2:
            success_counter_occupation += 1
        if pred_F72_2 == true_F72_2:
            success_counter_activity += 1
        if (pred_F71_2 == true_F71_2) & (pred_F72_2 == true_F72_2):
            success_counter_all += 1

        if pred_F71_2[:3] == true_F71_2[:3]:
            success_counter_occupation_3rd += 1
        if pred_F72_2[:3] == true_F72_2[:3]:
            success_counter_activity_3rd += 1

        if pred_F71_2[:2] == true_F71_2[:2]:
            success_counter_occupation_2nd += 1
        if pred_F72_2[:2] == true_F72_2[:2]:
            success_counter_activity_2nd += 1

        pred.at[index, "F71_1"] = row["F71_1"]
        pred.at[index, "F72_1"] = row["F72_1"]
        pred.at[index, "pred_F71_2"] = pred_F71_2
        pred.at[index, "pred_F72_2"] = pred_F72_2
        pred.at[index, "true_F71_2"] = true_F71_2
        pred.at[index, "true_F72_2"] = true_F72_2
        pred.at[index, "F73"] = F73_code
        pred.at[index, " pred_F71_cat"] = str(pred_ocu_cat)
        pred.at[index, " pred_F72_cat"] = str(pred_act_cat)
        pred.at[index, "Prob pred ocu"] = pred_ocu_prob
        pred.at[index, "Prob pred act"] = pred_act_prob
        pred.at[index, "Flag"] = flag

        if counter % 1000 == 0 and counter > 0:

            acc_por_tot_act = success_counter_activity / counter * 100
            acc_por_tot_ocu = success_counter_occupation / counter * 100
            acc_por_2nd_ocu = success_counter_occupation_2nd / counter * 100
            acc_por_2nd_act = success_counter_activity_2nd / counter * 100
            acc_por_3rd_ocu = success_counter_occupation_3rd / counter * 100
            acc_por_3rd_act = success_counter_activity_3rd / counter * 100

            print(
                "Van {} predicciones, con {} de aciertos totales, un {:.2f} %".format(
                    counter, success_counter_all, success_counter_all / counter * 100
                )
            )
            print(
                "Van {} predicciones, con {} de aciertos de actividad, un {:.2f}  , {:.2f}% al tercer nivel, {:.2f} % a la segundo nivel".format(
                    counter,
                    success_counter_activity,
                    acc_por_tot_act,
                    acc_por_3rd_act,
                    acc_por_2nd_act,
                )
            )
            print(
                "Van {} predicciones, con {} de aciertos de ocupacion, un {:.2f} %, {:.2f}% al tercer nivel, {:.2f} % a la segundo nivel".format(
                    counter,
                    success_counter_activity,
                    acc_por_tot_ocu,
                    acc_por_3rd_ocu,
                    acc_por_2nd_ocu,
                )
            )
    pred.to_excel(".\data\Predicciones.xlsx", index=False)
    print("Archivo guardado con exito")


if __name__ == "__main__":
    validate_models()
