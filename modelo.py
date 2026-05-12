import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Leer dataset limpio
df = pd.read_csv("data/infocana_limpio.csv")

# Variables independientes (X)
X = df[["cana_molida_neta"]]

# Variable dependiente (Y)
Y = df["azucar_producida_total"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Crear modelo
modelo = LinearRegression()

# Entrenar modelo
modelo.fit(X_train, y_train)

# Predicciones
predicciones = modelo.predict(X_test)

# Métricas
r2 = r2_score(y_test, predicciones)
mse = mean_squared_error(y_test, predicciones)

print("\n===== RESULTADOS =====")
print("R2:", r2)
print("MSE:", mse)

# Ecuación de la recta
print("\n===== ECUACIÓN =====")

print(
    f"Y = {modelo.intercept_} + "
    f"{modelo.coef_[0]}(X)"
)

# Graficar resultados
plt.figure(figsize=(10,6))

plt.scatter(
    X_test,
    y_test,
    label="Datos reales"
)

plt.plot(
    X_test,
    predicciones,
    color="red",
    linewidth=2,
    label="Predicción"
)

plt.xlabel("Caña Molida Neta")
plt.ylabel("Azúcar Producida Total")
plt.title("Regresión Lineal")

plt.legend()

plt.show()