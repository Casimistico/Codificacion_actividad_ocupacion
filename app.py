"""API for generating occupation and branch of activity codes from text"""


import json
import tempfile
import hug

from predictions import predict_code


@hug.get(
    "/predict_by_batch",
    output=hug.output_format.text,
    response_headers={"Content-Disposition": "attachment; filename=result.txt"},
)
def predict_by_batch(file):
    """
    Get occupation and activity codes by batch
    """
    # <body> is a simple dictionary of {filename: b'content'}

    file_desc, file_path = tempfile.mkstemp()
    with open(file, "r", encoding="utf8") as input_file:
        with open(file_desc, "w", encoding="utf8") as tmp_file:
            for line in input_file:
                data = line.strip().split(";")
                texto_ocupacion = data[0]
                texto_actividad = data[1]
                F73 = int(data[2])
                result = predict_code(texto_ocupacion, texto_actividad, F73)
                tmp_file.write(json.dumps(result,mensure_ascii=False))
                tmp_file.write("\n")
            tmp_file.close()
            input_file.close()
    f = open(file_path, "rb")
    return f


@hug.get("/predict_codes")
def predict_codes(
    texto_ocupacion: hug.types.text,
    texto_actividad: hug.types.text,
    F73: hug.types.text,
):
    """endpoint for simple case"""

    occupation_result, activity_result, flag = predict_code(
        texto_ocupacion, texto_actividad, int(F73)
    )

    data = {
        "Código CIUO08:": occupation_result[0],
        "Precisión de entrenamiento código ocupación:": occupation_result[2],
        "Descripción ocupación:": occupation_result[1],
        "Código CNUO95:": activity_result[0],
        "Precisión de entrenamiento rama actividad:": activity_result[2],
        "Descripción rama actividad:": activity_result[1],
        "Comentarios:": flag,
    }

    return {"data": data}
