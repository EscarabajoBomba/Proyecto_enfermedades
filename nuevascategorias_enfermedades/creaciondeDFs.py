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

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')
#Comienzo a crear los dataframes filtrados por meses tipo Ene-Feb ...
#dataframefinal = pd.read_csv('dataframefinal.csv', sep = ',')
archivo = 'DFmenores.csv'
dataframefinal =  pd.read_csv(archivo, sep = ',')  #con este comienzo el tratamiento final por mes

## aqui renomambro el resto de columnas

dataframefinal.rename(columns = {'-_COVID_19_CONFIRMADO_(U07.1)':'COVID19_Confirmado_h'}, inplace = True)

#enfermedades.rename(columns = {'mp25':'MP2.5'}, inplace = True)
#elimino ultimo row
dataframefinal.drop(dataframefinal.tail(1).index, inplace = True)
#aqui comienzo a eliminar columnas que no me sirven
del dataframefinal["Edad_y_Tipo_de_Atención"]
#del dataframefinal["Unnamed: 0.1"]
del dataframefinal["Unnamed: 0"]
del dataframefinal["Covid-19_Virus_no_identificado_U07.2"]
del dataframefinal["Covid-19_Virus_identificado_U07.1"]
del dataframefinal["-_COVID-19_VIRUS_NO_IDENTIFICADO_U07.2"]
del dataframefinal["-_COVID-19_VIRUS_IDENTIFICADO_U07.1"]
del dataframefinal["COVID19_Sospechoso_u"]
del dataframefinal["COVID19_Sospechoso_h"]


print(dataframefinal.info())
##aqui reemplazo todos los Nan por un 0
dataframefinal = dataframefinal.fillna(0)
#dataframefinal.to_csv('dataframefinal.csv') #aqui reescribo el dataframe

#enfermedades.to_csv('DF/dataframe.csv')
print('----------------------')
print('Media de cada variable')
print('----------------------')
print(dataframefinal.mean(axis=0))

print('-------------------------')
print('Varianza de cada variable')
print('-------------------------')
print(dataframefinal.var(axis=0))

def extraer_meses(data):
    ene_feb = data[0:8]
    ene_feb = ene_feb.append(data[52:60])
    ene_feb = ene_feb.append(data[104:112])
    ene_feb = ene_feb.append(data[156:164])
    ene_feb = ene_feb.append(data[209:217])

    mar_abr = data[8:17]
    mar_abr = mar_abr.append(data[60:69])
    mar_abr = mar_abr.append(data[112:121])
    mar_abr = mar_abr.append(data[164:173])
    mar_abr = mar_abr.append(data[217:226])

    may_jun = data[17:25]
    may_jun = may_jun.append(data[69:77])
    may_jun = may_jun.append(data[121:129])
    may_jun = may_jun.append(data[173:181])
    may_jun = may_jun.append(data[226:234])

    jul_ago = data[25:34]
    jul_ago = jul_ago.append(data[77:86])
    jul_ago = jul_ago.append(data[129:138])
    jul_ago = jul_ago.append(data[181:190])
    jul_ago = jul_ago.append(data[234:243])

    sep_oct = data[34:43]
    sep_oct = sep_oct.append(data[86:95])
    sep_oct = sep_oct.append(data[138:147])
    sep_oct = sep_oct.append(data[190:199])
    sep_oct = sep_oct.append(data[243:252])

    nov_dic = data[43:52]
    nov_dic = nov_dic.append(data[95:104])
    nov_dic = nov_dic.append(data[147:156])
    nov_dic = nov_dic.append(data[199:208])
    nov_dic = nov_dic.append(data[252:])

    return ene_feb, mar_abr, may_jun, jul_ago, sep_oct, nov_dic

primer, segundo, tercero, cuarto, quinto ,sexto = extraer_meses(dataframefinal) 
primer.to_csv('porfechayrango/ene_feb'+ archivo)
segundo.to_csv('porfechayrango/mar_abr'+ archivo)
tercero.to_csv('porfechayrango/may_jun'+ archivo)
cuarto.to_csv('porfechayrango/jul_ago'+ archivo)
quinto.to_csv('porfechayrango/sep_oct'+ archivo)
sexto.to_csv('porfechayrango/nov_dic'+ archivo)


#print(primer, segundo, tercero, cuarto, quinto ,sexto)
