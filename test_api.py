import json
import requests
import pandas as pd
import pickle
from utiles import *

# Ajustamos los encabezados para enviar y aceptar respuestas JSON

header = {'Content-Type': 'application/json', 'Accept':'application/json'}

#Creamos un dataset de pruebas
df = pd.DataFrame({'unidades':[289,288,260,240,290,255,270,300],
                    'weekday':[5,0,1,2,3,4,5,0],
                    'month':[4,4,4,4,4,4,4,4]})
loader_scaler = load_object('scaler_time_series.pkl')

reframed = transformar(df, loader_scaler)

reordenado = reframed[['weekday', 'month','var1(t-7)','var1(t-6)','var1(t-5)','var1(t-4)','var1(t-3)','var1(t-2)','var1(t-1)']]
reordenado.dropna(inplace=True)

# Convertimos DATAFRAME de pandas a JSON
data = reordenado.to_json(orient='records')

print('JSON para enviar en POST', data)

# Aquí le hacemos POST hacia la URL = PREDICT

resp = requests.post("http://127.0.0.1:5000/predict", data = json.dumps(data), headers= header)

print('status',resp.status_code)

# La respuesta final la obtenemos así:
print('Respuesta de servidor')
print(resp.json())