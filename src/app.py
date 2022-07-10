import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf
import seaborn as sns 


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Example importing the CSV here
#Step 1
df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv')

pd.set_option('display.max_columns',110)

df.describe()

#mini encuesta de hogares
df.info()
df.sample(10)
##hacemos la lista de variable
list(df.columns)
#Hacemos el EDA para ver q hacemos
#Step 2
#armo un correlation matrix   , mapa de calor
plt.figure(figsize=(25,25))
corr_matrix = df.corr()
hit_map = sns.heatmap(corr_matrix,annot=False)
plt.show()

#buscamos zonas bien ocuras las negras, son correlaciones negativas altas
#las claras son correlaciones positivas altas
#vemos q grupo 
#elegimos una variable, Camila nos pasa un codigo, vemos la var y aplicamos LASO
print(corr_matrix)
df_heart_disease = pd.DataFrame(df.corrwith(df['Heart disease_prevalence'],axis=0),columns=['Correlacion'
#filtramos y vemos los q son mayores a X numero, para establecer cierto umbral

df_heart_disease[abs(df_heart_disease['Correlacion']) > 0.8] #esto nos da las vars q vamos a sacar del modelo, las dropeamos
df_heart_disease[abs(df_heart_disease['Correlacion']) > 0.8].index #la lista de las vars a elminiar
x = df.drop(['Heart disease_prevalence', 'Heart disease_Lower 95% CI',
       'Heart disease_Upper 95% CI', 'COPD_prevalence', 'COPD_Lower 95% CI',
       'COPD_Upper 95% CI', 'diabetes_prevalence', 'diabetes_Lower 95% CI',
       'diabetes_Upper 95% CI', 'CKD_prevalence', 'CKD_Lower 95% CI',
       'CKD_Upper 95% CI'],axis=1)
y = df['Heart disease_prevalence']

x.sample(5)
df.shape #veo las vars q quedaron
#3140 observaciones en 108 columnas
df.describe(include='O')
x = x.drop(['COUNTY_NAME'], axis=1)
x = pd.get_dummies(x,drop_first=True) #toma vars categoricas y las pasa a dummies. los string los encodeo en 0 y 1

#scamos el county name porq es mucho agregar 1841 categorias lo podria dejar para otro estudio, ahora las elimino
x.sample(5)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=53, test_size=0.15)
#corremos las regresion Laso, para ver q var nos quedan, esto saca las otras var q no son redundantes para el modelo
modelo = Lasso(alpha = 0.3,normalize = True)
modelo.fit(X_train,y_train)
predicciones = modelo.predict(X_test)
rmse_lasso = mean_squared_error(
y_true = y_test,
y_pred = predicciones,
squared = False
)
print("")
print(f"El error (rmse) de test es: {rmse_lasso}")
# Creación y entrenamiento del modelo (con búsqueda por CV del valor óptimo alpha)
# ==============================================================================
# Por defecto LassoCV utiliza el mean squared error
modelo = LassoCV(
alphas = np.logspace(-10, 3, 200),
normalize = True,
cv = 10
)
_ = modelo.fit(X = X_train, y = y_train)
# Evolución de los coeficientes en función de alpha
# ==============================================================================
alphas = modelo.alphas_
coefs = []

for alpha in alphas:
    modelo_temp = Lasso(alpha=alpha, fit_intercept=False, normalize=True)
    modelo_temp.fit(X_train, y_train)
    coefs.append(modelo_temp.coef_.flatten())

fig, ax = plt.subplots(figsize=(7, 3.84))
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_ylim([-15,None])
ax.set_xlabel('alpha')
ax.set_ylabel('coeficientes')
ax.set_title('Coeficientes del modelo en función de la regularización')
# Número de predictores incluidos (coeficiente !=0) en función de alpha
# ==============================================================================
alphas = modelo.alphas_
n_predictores = []

for alpha in alphas:
    modelo_temp = Lasso(alpha=alpha, fit_intercept=False, normalize=True)
    modelo_temp.fit(X_train, y_train)
    coef_no_cero = np.sum(modelo_temp.coef_.flatten() != 0)
    n_predictores.append(coef_no_cero)

fig, ax = plt.subplots(figsize=(7, 3.84))
ax.plot(alphas, n_predictores)
ax.set_xscale('log')
ax.set_ylim([-15,None])
ax.set_xlabel('alpha')
ax.set_ylabel('nº predictores')
ax.set_title('Predictores incluidos en función de la regularización');
# Evolución del error en función de alpha
# ==============================================================================
# modelo.mse_path_ almacena el mse de cv para cada valor de alpha. Tiene
# dimensiones (n_alphas, n_folds)

mse_cv = modelo.mse_path_.mean(axis=1)
mse_sd = modelo.mse_path_.std(axis=1)

# Se aplica la raíz cuadrada para pasar de mse a rmse
rmse_cv = np.sqrt(mse_cv)
rmse_sd = np.sqrt(mse_sd)

# Se identifica el óptimo y el óptimo + 1std
min_rmse = np.min(rmse_cv)
sd_min_rmse = rmse_sd[np.argmin(rmse_cv)]
optimo = modelo.alphas_[np.argmin(rmse_cv)]

# Gráfico del error +- 1 desviación estándar
fig, ax = plt.subplots(figsize=(7, 3.84))
ax.plot(modelo.alphas_, rmse_cv)
ax.fill_between(
modelo.alphas_,
rmse_cv + rmse_sd,
rmse_cv - rmse_sd,
alpha=0.2
)

ax.axvline(
x = optimo,
c = "gray",
linestyle = '--',
label = 'óptimo'
)



ax.set_xscale('log')
ax.set_ylim([0,None])
ax.set_title('Evolución del error CV en función de la regularización')
ax.set_xlabel('alpha')
ax.set_ylabel('RMSE')
plt.legend()
#grafica q mide el error con los dif alfas, vemos el alfa optimo es 
print(f"Mejor valor de alpha encontrado: {modelo.alpha_}")
# Mejor valor alpha encontrado + 1sd
# ==============================================================================
min_rmse     = np.min(rmse_cv)
sd_min_rmse  = rmse_sd[np.argmin(rmse_cv)]
min_rsme_1sd = np.max(rmse_cv[rmse_cv <= min_rmse + sd_min_rmse])
optimo       = modelo.alphas_[np.argmin(rmse_cv)]
optimo_1sd   = modelo.alphas_[rmse_cv == min_rsme_1sd]

print(f"Mejor valor de alpha encontrado + 1 desviación estándar: {optimo_1sd}")
# Mejor modelo alpha óptimo + 1sd
# ==============================================================================

modelo = Lasso(alpha= 0.00113573, normalize=True)
modelo.fit(X_train, y_train)
# Coeficientes del modelo
# ==============================================================================
df_coeficientes = pd.DataFrame(
                        {'predictor': X_train.columns,
                         'coef': modelo.coef_.flatten()}
                  )

# Predictores incluidos en el modelo (coeficiente != 0)
df_coeficientes[df_coeficientes.coef != 0]
fig, ax = plt.subplots(figsize=(11, 3.84))
ax.stem(df_coeficientes.predictor, df_coeficientes.coef, markerfmt=' ')
plt.xticks(rotation=90, ha='right', size=5)
ax.set_xlabel('variable')
ax.set_ylabel('coeficientes')
ax.set_title('Coeficientes del modelo');
# Predicciones test
# ==============================================================================
predicciones = modelo.predict(X=X_test)
predicciones = predicciones.flatten()
predicciones[:10]
# Error de test del modelo 
# ==============================================================================
rmse_lasso = mean_squared_error(
                y_true  = y_test,
                y_pred  = predicciones,
                squared = False
             )
print("")
print(f"El error (rmse) de test es: {rmse_lasso}")

