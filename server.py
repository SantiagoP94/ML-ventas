import pandas as pd
from flask import Flask, jsonify, request

from utiles import *

app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def predict():
    #API REQUEST
    try:
        req_json = request.get_json()
        return (req_json)
        input = pd.read_json(req_json, orient='records')
    except Exception as e:
        raise e
    
    if input.empty:
        return('NO PAGINA')
    else:
        #Cargar el modelo guardado
        print('Cargar modelo...')
        loaded_model = cargarModeloSiEsNecesario()

        print('Realizar Pronosticos')
        continuas = input[['var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)']]
        predictions = loaded_model.predict([input['weekday'], input['month'], continuas])

        print('transformando datos')
        loaded_scaler = load_object('scaler_time_series.pkl')
        inverted = loaded_scaler.inverse_transform(predictions)
        inverted = inverted.astype('int32')

        final_predictions = pd.DataFrame(inverted)
        final_predictions.columns = ['ventas']

        print('enviar respuesta')
        responses = jsonify(predictions=final_predictions.to_json(orient='records'))
        responses.status_code=200
        print('fin peticion')

        return (responses)
    

@app.route('/', methods=['GET', 'POST'])
def index():    
    if request.method=='POST':
        print('POST')
    else:
        print('GET')
    return('hola')

#Cargar modelo si es necesario
global_model = None

def cargarModeloSiEsNecesario():
    global global_model
    if global_model is not None:
        print('Modelo ya cargado')
        return global_model
    else:
        global_model = crear_modeloEmbeddings()
        global_model.load_weights("pesos.h5")
        print('Modelo Cargado')
        return global_model

if __name__ == '__main__':
    app.run(debug=True)
