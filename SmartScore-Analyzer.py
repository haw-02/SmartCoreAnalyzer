import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

df = pd.read_csv("C:\\Users\\User\\OneDrive - Instituto Tecnológico de Las Américas (ITLA)\\Samsung certificacion\\student_exam_scores.csv")

df.head(10)

print("\n=== INFORMACION GENERAL ===")
print(df.info())
print("\nDimensiones del dataset:")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")


print("\n=== VALORES FALTANTES ===")
print(df.isnull().sum())
print("\nNumero de filas duplicadas:", df.duplicated().sum())



print("\n=== TIPOS DE VARIABLES ===")
print(df.dtypes)


print("\n=== ESTADISTICAS DESCRIPTIVAS ===")
print(df.describe(include='all').T)


### Medidas adicionales de dispersion y forma
numeric_cols = df.select_dtypes(include=np.number).columns

print("\n=== MEDIDAS ADICIONALES ===")
for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  Media: {df[col].mean():.2f}")
    print(f"  Mediana: {df[col].median():.2f}")
    print(f"  Moda: {df[col].mode()[0]}")
    print(f"  Desviación estándar: {df[col].std():.2f}")
    print(f"  Varianza: {df[col].var():.2f}")
    print(f"  Mínimo: {df[col].min()}")
    print(f"  Máximo: {df[col].max()}")
    print(f"  Rango: {df[col].max() - df[col].min():.2f}")
    print(f"  Coef. de asimetría: {df[col].skew():.2f}")
    print(f"  Curtosis: {df[col].kurtosis():.2f}")



numeric_cols = df.select_dtypes(include=np.number).columns

print("\n=== MEDIDAS ADICIONALES ===")
for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  Media: {df[col].mean():.2f}")
    print(f"  Mediana: {df[col].median():.2f}")
    print(f"  Moda: {df[col].mode()[0]}")
    print(f"  Desviación estándar: {df[col].std():.2f}")
    print(f"  Varianza: {df[col].var():.2f}")
    print(f"  Mínimo: {df[col].min()}")
    print(f"  Máximo: {df[col].max()}")
    print(f"  Rango: {df[col].max() - df[col].min():.2f}")
    print(f"  Coef. de asimetría: {df[col].skew():.2f}")
    print(f"  Curtosis: {df[col].kurtosis():.2f}")


for col in numeric_cols:
  plt.figure(figsize=(8,4))
  sns.histplot(df[col], kde=True, bins=20)
  plt.title(f"Distribucion de {col}")
  plt.xlabel(col)
  plt.ylabel("Frecuencia")
  plt.show()



for col in numeric_cols:
  plt.figure(figsize=(6,3))
  sns.boxplot(x=df[col])
  plt.title(f"Boxplot de {col}")
  plt.show()


plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlacion")
plt.show()

sns.pairplot(df, diag_kind="kde")
plt.suptitle("Relaciones entre variables numéricas", y=1.02)
plt.show()


print("\n=== RESUMEN DEL EDA ===")
print("No. de filas y columnas:", df.shape)
print("Columnas numericas:", list(numeric_cols))
print("Columnas categoricas:", list(df.select_dtypes(exclude=np.number).columns))
print("Valores faltantes totales:", df.isnull().sum().sum())
print("Correlacion mas alta:")
corr_matrix = df[numeric_cols].corr().abs()
np.fill_diagonal(corr_matrix.values, np.nan)
max_corr = corr_matrix.unstack().idxmax()
print(f"   {max_corr} con correlación {corr_matrix.unstack().max():.2f}")


mejores5 = df.sort_values(by="hours_studied", ascending=False).head(5)
print("los 5 que más estudiaron:\n", mejores5[["student_id", "hours_studied", "exam_score"]], "\n")

peores5 = df.sort_values(by="hours_studied", ascending=True).head(5)
print("los 5 que menos estudiaron:\n", peores5[["student_id", "hours_studied", "exam_score"]], "\n")

promedioMasEstudiarion = mejores5["exam_score"].mean()
promedioMenosEstudiaron = peores5["exam_score"].mean()
print(f"promedio nota que mas estudiaron: {promedioMasEstudiarion:.2f}")
print(f"promedio nota que menos estudiaron: {promedioMenosEstudiaron:.2f}\n")

plt.bar(mejores5["student_id"], mejores5["exam_score"], color="green", label="Más estudiaron")
plt.bar(peores5["student_id"], peores5["exam_score"], color="red", label="Menos estudiaron")
plt.xlabel("ID del estudiante")
plt.ylabel("nota del examen")
plt.title("notas: mas vs menos horas de estudio")
plt.legend()
plt.show()

print("los que más estudiaron sacaron mejores notas en promedio.")


print(f"\n{'='*50}")
print("INTERPRETACION:")
print(f"{'='*50}")

if corr > 0.7:
    interpretacion = "Hay una relación positiva FUERTE: a mayor asistencia, significativamente mayores notas."
elif corr > 0.5:
    interpretacion = "Hay una relación positiva MODERADA-FUERTE: a mayor asistencia, mayores notas."
elif corr > 0.3:
    interpretacion = "Relación positiva MODERADA: asistir más puede ayudar considerablemente."
elif corr > 0.1:
    interpretacion = "Relación positiva DÉBIL: asistir más influye ligeramente."
elif corr > -0.1:
    interpretacion = "NO hay relación clara entre asistencia y nota."
elif corr > -0.3:
    interpretacion = "Relación negativa DÉBIL (poco común en este contexto)."
else:
    interpretacion = "Relación negativa: más asistencia asociada a menores notas (situación inusual)."

print(interpretacion)
print(f"\nLa asistencia explica aproximadamente el {r_squared*100:.1f}% de la")
print(f"variabilidad en las notas de los exámenes.")
print(f"\n{'='*50}")
print("MATRIZ DE CORRELACION COMPLETA:")
print(f"{'='*50}")
correlation_matrix = df.corr(numeric_only=True)
correlation_matrix_rounded = correlation_matrix.round(3)
print(correlation_matrix_rounded)

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(correlation_matrix_rounded, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

ax.set_xticks(range(len(correlation_matrix_rounded.columns)))
ax.set_yticks(range(len(correlation_matrix_rounded.columns)))
ax.set_xticklabels(correlation_matrix_rounded.columns, rotation=45, ha='right')
ax.set_yticklabels(correlation_matrix_rounded.columns)

ax.set_title('Matriz de Correlación - Mapa de Calor', fontsize=14, fontweight='bold', pad=20)

plt.colorbar(cax, ax=ax, label='Coeficiente de Correlación')

for i in range(len(correlation_matrix_rounded)):
    for j in range(len(correlation_matrix_rounded)):
        value = correlation_matrix_rounded.iloc[i, j]
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='black', fontsize=10)

plt.tight_layout()
plt.show()


print("=" * 50)
print("ANALISIS DE CORRELACION: ASISTENCIA vs NOTAS")
print("=" * 50)
print(f"\nNumero de estudiantes: {len(df)}")
print(f"\nEstadisticas de Asistencia:")
print(df["attendance_percent"].describe())
print(f"\nEstadísticas de Notas:")
print(df["exam_score"].describe())


x = df["attendance_percent"]
y = df["exam_score"]

corr = x.corr(y)
print(f"\n{'='*50}")
print(f"Coeficiente de Correlacion de Pearson: {corr:.4f}")
print(f"{'='*50}")

r_squared = corr ** 2
print(f"R al cuadrado (varianza explicada): {r_squared:.4f} ({r_squared*100:.2f}%)")

pearson_stat, p_value = stats.pearsonr(x, y)
print(f"P-valor: {p_value:.4e}")
if p_value < 0.05:
    print("✓ La correlacion es estadisticamente significativa (p < 0.05)")
else:
    print("✗ La correlacion NO es estadísticamente significativa (p ≥ 0.05)")

# Ajuste lineal 
m, b = np.polyfit(x, y, 1)
print(f"\nEcuacion de la recta: y = {m:.3f}x + {b:.3f}")
print(f"Interpretación: Por cada 1% de aumento en asistencia,")
print(f"la nota aumenta aproximadamente {m:.3f} puntos")


plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='steelblue', alpha=0.6, s=50, 
            edgecolors='navy', linewidth=0.5, label='Estudiantes')
plt.plot(x, m*x + b, color='red', linewidth=2.5, 
         label=f'Tendencia lineal (r={corr:.3f})')
plt.xlabel("Asistencia (%)", fontsize=12, fontweight='bold')
plt.ylabel("Nota del examen", fontsize=12, fontweight='bold')
plt.title("Relacion entre Asistencia y Nota del Examen", 
          fontsize=14, fontweight='bold', pad=20)
plt.show()


corr = df["previous_scores"].corr(df["exam_score"])
print(f"Correlacion entre nota anterior y nueva: {corr:.3f}")

df_sorted = df.sort_values("previous_scores", ascending=False).head(20) 
plt.figure(figsize=(10,6))
bar_width = 0.35
indices = np.arange(len(df_sorted))
plt.bar(indices, df_sorted["previous_scores"], bar_width, label="Nota anterior", color='skyblue')
plt.bar(indices + bar_width, df_sorted["exam_score"], bar_width, label="Nota nueva", color='salmon')
plt.xlabel("Estudiantes (Top 20 por nota anterior)")
plt.ylabel("Nota")
plt.title("Comparacion entre nota anterior y nota nueva")
plt.xticks(indices + bar_width / 2, df_sorted.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


if corr > 0.7:
    print("Sí: los buenos estudiantes suelen mantener su rendimiento (correlacion fuerte).")
elif corr > 0.4:
    print("Parcialmente: hay cierta tendencia, pero con variación (correlacion moderada).")
elif corr > 0.1:
    print("Débil: algunos buenos estudiantes se mantienen, otros bajan o suben mucho.")
else:
    print("No: las notas pasadas no predicen las nuevas (correlacion muy baja).")