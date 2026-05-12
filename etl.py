import pandas as pd

# Leer CSV
df = pd.read_csv("data/infocana.csv")

# Mostrar columnas
print(df.columns)

# Limpiar nombres
df.columns = df.columns.str.strip().str.lower()

# Eliminar valores nulos
df = df.dropna()

# Convertir columnas numéricas
df["cana_molida_neta"] = pd.to_numeric(
    df["cana_molida_neta"],
    errors="coerce"
)

df["azucar_producida_total"] = pd.to_numeric(
    df["azucar_producida_total"],
    errors="coerce"
)

# Eliminar nulos después de convertir
df = df.dropna()

# Guardar dataset limpio
df.to_csv("data/infocana_limpio.csv", index=False)

# Mostrar primeras filas
print(df.head())

print("\nETL completado correctamente")