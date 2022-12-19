import csv
import datetime

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado
# ==============================================================================
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

dataframefinal = pd.read_csv('dataframefinal.csv', sep = ',')

## aqui renomambro el resto de columnas

#enfermedades.rename(columns = {'mp25':'MP2.5'}, inplace = True)
#elimino ultimo row
dataframefinal.drop(dataframefinal.tail(1).index, inplace = True)
#aqui comienzo a eliminar columnas que no me sirven
#del dataframefinal["Edad_y_Tipo_de_Atención"]
del dataframefinal["Unnamed: 0.1"]
del dataframefinal["Unnamed: 0"]
#del dataframefinal["Covid-19_Virus_no_identificado_U07.2"]
#del dataframefinal["Covid-19_Virus_identificado_U07.1"]
#del dataframefinal["-_COVID-19_VIRUS_NO_IDENTIFICADO_U07.2"]
#del dataframefinal["-_COVID-19_VIRUS_IDENTIFICADO_U07.1"]
#del dataframefinal["COVID19_Sospechoso_u"]
#del dataframefinal["COVID19_Sospechoso_h"]


print(dataframefinal.info())
##aqui reemplazo todos los Nan por un 0
dataframefinal = dataframefinal.fillna(0)
dataframefinal.to_csv('dataframefinal.csv') #aqui reescribo el dataframe

#enfermedades.to_csv('DF/dataframe.csv')
print('----------------------')
print('Media de cada variable')
print('----------------------')
print(dataframefinal.mean(axis=0))

print('-------------------------')
print('Varianza de cada variable')
print('-------------------------')
print(dataframefinal.var(axis=0))

# Entrenamiento modelo PCA con escalado de los datos
# ==============================================================================
pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(dataframefinal)

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']
print(modelo_pca.components_)

# Se combierte el array a dataframe para añadir nombres a los ejes.
datapca = pd.DataFrame(
    data    = modelo_pca.components_,
    columns = dataframefinal.columns,
    index   = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18']
)
print(datapca)

# Heatmap componentes
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
componentes = modelo_pca.components_
plt.imshow(componentes.T, cmap='viridis', aspect='auto')
plt.yticks(range(len(dataframefinal.columns)), dataframefinal.columns)
plt.xticks(range(len(dataframefinal.columns)), np.arange(modelo_pca.n_components_) + 1)
plt.grid(False)
plt.colorbar();
plt.show()

# Porcentaje de varianza explicada por cada componente
# ==============================================================================
print('----------------------------------------------------')
print('Porcentaje de varianza explicada por cada componente')
print('----------------------------------------------------')
print(modelo_pca.explained_variance_ratio_)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(
    x      = np.arange(modelo_pca.n_components_) + 1,
    height = modelo_pca.explained_variance_ratio_
)

for x, y in zip(np.arange(len(dataframefinal.columns)) + 1, modelo_pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Porcentaje de varianza explicada por cada componente')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza explicada');
plt.show()


# Porcentaje de varianza explicada acumulada
# ==============================================================================
prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
print('------------------------------------------')
print('Porcentaje de varianza explicada acumulada')
print('------------------------------------------')
print(prop_varianza_acum)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.plot(
    np.arange(len(dataframefinal.columns)) + 1,
    prop_varianza_acum,
    marker = 'o'
)

for x, y in zip(np.arange(len(dataframefinal.columns)) + 1, prop_varianza_acum):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

ax.set_ylim(0, 1.1)
ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_title('Porcentaje de varianza explicada acumulada')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza acumulada');
plt.show()

# Proyección de las observaciones de entrenamiento
# ==============================================================================
proyecciones = pca_pipe.transform(X=dataframefinal)
proyecciones = pd.DataFrame(
    proyecciones,
    columns =  ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18'],
    index   = dataframefinal.index
)
proyecciones.head()

proyecciones = np.dot(modelo_pca.components_, scale(dataframefinal).T) # (dataframefinal).T
proyecciones = pd.DataFrame(proyecciones, index = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18'])
proyecciones = proyecciones.transpose().set_index(dataframefinal.index)
proyecciones.head()

print(proyecciones.head())
## intento de plotear el dataframe como tabla
## LAS PROXIMAS 5 lineas son para la tabla

#fig, ax = plt.subplots()
# hide axes
#fig.patch.set_visible(False)
#ax.axis('off')
#ax.axis('tight')
#ax.table(cellText=enfermedades.values, colLabels=enfermedades.columns, loc='center')

#fig.tight_layout()

dataframefinal['Neumonia'].plot(kind = 'bar')
plt.show()
