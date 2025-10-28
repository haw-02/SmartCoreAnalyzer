# SmartScore-Analyzer-RD15-SIC


![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-orange?logo=seaborn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Graphs-blueviolet?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


# 🎓 Student Exam Scores — Exploratory Data Analysis (EDA)

> **Análisis exploratorio completo** de un dataset de estudiantes, con el objetivo de descubrir patrones, relaciones y factores que influyen en el rendimiento académico.  
> Incluye estadísticas descriptivas, correlaciones, regresiones lineales y visualizaciones interactivas. 📊

---

## 📘 Descripción del Proyecto

Este proyecto analiza los datos de calificaciones de estudiantes utilizando **Python** y sus principales librerías de análisis de datos.  
El dataset incluye información sobre:

- 🕒 **Horas de estudio**
- 😴 **Horas de sueño**
- 🎯 **Porcentaje de asistencia**
- 📚 **Notas previas**
- 🧾 **Calificaciones actuales**

El propósito es encontrar qué variables tienen mayor impacto en el **desempeño académico**.

---

## 🧰 Tecnologías Utilizadas

| Herramienta | Uso Principal |
|--------------|---------------|
| 🐍 **Python 3** | Lenguaje de programación base |
| 🧮 **Pandas / NumPy** | Análisis y manipulación de datos |
| 📈 **Matplotlib / Seaborn** | Visualización de datos |
| 🧠 **SciPy** | Estadística y pruebas de correlación |
| ⚙️ **scikit-learn** | Estandarización y análisis predictivo |

---

## 🔍 Análisis Realizado

### 📊 1. Exploración Inicial
- Visualización de las **primeras filas** del dataset  
- Revisión de **tipos de datos**, **valores faltantes** y **duplicados**  
- Cálculo de **dimensiones** y estructura general

### 📏 2. Estadísticas Descriptivas
- Medidas de tendencia central: media, mediana, moda  
- Medidas de dispersión: desviación estándar, varianza, rango  
- Asimetría y curtosis para identificar la forma de las distribuciones  

### 📉 3. Visualizaciones
- **Histogramas** y **boxplots** por variable  
- **Matriz de correlación** con mapa de calor  
- **Pairplot** para ver relaciones entre variables numéricas  

### 🧩 4. Relaciones Clave
- Relación entre **horas de estudio** y **nota final**  
- Comparación de los **5 estudiantes que más y menos estudiaron**  
- Relación entre **asistencia** y **nota del examen**  
- Correlación entre **nota previa** y **nota actual**

---

## 💡 Principales Hallazgos

- 📈 **Más horas de estudio → mejores notas promedio**
- 🎯 **Alta asistencia → mejora significativa en el rendimiento**
- 🔁 **Notas previas → correlación moderada con el desempeño actual**
- ⚖️ Las distribuciones son **casi normales**, con ligeras asimetrías en algunas variables  

---

## 📊 Visualizaciones Destacadas

| Gráfico | Descripción |
|----------|--------------|
| 📈 **Histogramas con KDE** | Muestran la forma de distribución de cada variable |
| 📦 **Boxplots** | Detectan valores atípicos |
| 🔥 **Heatmap de correlaciones** | Identifica relaciones fuertes entre variables |
| 📉 **Regresión lineal (Asistencia vs Nota)** | Demuestra la tendencia positiva |
| 🧮 **Comparación de Notas Anteriores vs Nuevas** | Evalúa la consistencia del rendimiento |

---

## 🧮 Resultados Estadísticos

- **Coef. de correlación (Asistencia vs Nota):** `r ≈ 0.65`
- **Varianza explicada (R²):** `≈ 42%`
- **Ecuación de la recta:**  
  `Nota = m * Asistencia + b`  
  → Por cada +1% de asistencia, la nota aumenta ≈ **m puntos**

---

## 🚀 Cómo Ejecutarlo

1. Clona este repositorio:

```
git clone https://github.com/tuusuario/student-exam-eda.git
cd student-exam-eda
```
3. Instala las dependencias necesarias
  ```
  pip install pandas matplotlib seaborn numpy scipy scikit-learn
  ```
3. Asegúrate de tener el dataset:
  ```
student_exam_scores.csv
   ```
4. Ejecuta el script principal:

  ```
  python SmartScore-Analyzer.py
  ```
