import pickle
import pandas as pd

EPOCHS = 40
PASOS=7

def save_object(filename, object):
    with open(''+filename, 'wb') as file:
        pickle.dump(object, file)
def load_object(filename):
    with open (''+filename, 'rb') as f:
        loaded = pickle.load(f)
    return loaded

#FUNCION PARA convertir SERIESTime en APRENDIZAJE SUPERVISADO
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars= 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    #Secuencia de entrada (t-n, ... t-1)
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names+=[('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    #Secuencia de pronosticos(t,t+1,...t+n).
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1,i)) for j in range(n_vars)]
    #Concatenar todo(Poner todos juntos)
    agg=pd.concat(cols,axis=1)
    agg.columns = names
    #Eliminar valores con NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#Funcion para CREAR EL MODELO EMBEDDINGS(RED NEURONAL).
def crear_modeloEmbeddings():
    from keras.models import Sequential
    from keras.layers import Activation, Input, Embedding, Dense, Flatten, Dropout,concatenate,LSTM
    from keras.layers import BatchNormalitazion, SpatialDropout1D
    from keras.callbacks import Callback
    from keras.models import Model, load_model
    from keras.optimizers import Adam
    
    emb_dias = 2 #Tamaño de profundidad de los embeddings
    emb_meses = 4
    
    in_dias = Input(shape = [1,], name = 'dias')
    emb_dias = Embedding(7+1,emb_dias)(in_dias)
    in_meses = Input(shape=[1,], name = 'meses')
    emb_meses = Embedding(12+1, emb_meses)(in_meses)
    in_cli=Input(shape=[PASOS,], name='cli')
    fe=concatenate([(emb_dias), (emb_meses)])
    x=Flatten()(fe)
    x = Dense(PASOS, activation='tanh')(x)
    outp = Dense(1,activation='tanh')(x)
    model = Model(inputs=[in_dias,in_meses,in_cli], outputs=outp)
    model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['MSE'])
    return model

def transformar(df,scaler):
    #Cargar valores
    values = df['unidades'].values
    #Pasar a tipo FLOAT
    values = values.astype('float32')
    #Normalizar features(caracteristicas)
    values = values.reshape(-1,1) #Esto lo hacemos porque tenemos 1 sola dimension
    scaled = scaler.fit_transform(values)
    
    reframed = series_to_supervised(scaled,PASOS,1)
    reframed.reset_index(inplace=True, drop=True)
    
    contador = 0
    reframed['weekday']=df['weekday']
    reframed['month']=df['month']
    
    for i in range(reframed.index[0], reframed.index[-1]):
        reframed['weekday'].loc[contador]=df['weekday'][i+8]
        reframed['month'].loc[contador]=df['month'][i+8]
        contador+=1
    return reframed


