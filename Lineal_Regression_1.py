print(__doc__)

# Code source: Jaques Grobler
# License: BSD 3 clause
# Modificado por: Fede
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Importamos la database de "diabetes"
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Dividimos los datos en conjuntos de entrenamiento y test
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Dividimos los objetivos en conjuntos de entrenamiento y test
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Creamos el objeto de regresion lineal
regr = linear_model.LinearRegression()

# Entrenamos el modelo
regr.fit(diabetes_X_train, diabetes_y_train)

# Realizamos las predicciones
diabetes_y_pred = regr.predict(diabetes_X_test)

# Coeficientes
print('Coeficientes: \n', regr.coef_)
print('hpla')
# Error medio al cuadrado
print('Error medio al cuadrado: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Coeficiente de determinacion: 1 es perfecto
print('Coeficiente de determinacion: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Salida de gráficas
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()