Proyecto predicción de enfermedad cardíaca.
Facultad de Ciencias, UNAM
Profesores: David Alexis García Espinosa, Derek Saúl Morán Pérez, Luis Eduardo Flores Luna
Integrantes: Admin Zaid Ibáñez Martínez, Jesús Eduardo Guerra Crespo.
Temas Selectos de Biomatemáticas: Introducción a la ciencia de Datos Aplicada a escenarios médico-biológicos
Grupo 8416
INSTRUCCIONES PARA USAR EL PROYECTO DE PREDICCIÓN DE ENFERMEDAD CARDÍACA

Descripción General Este proyecto realiza Análisis Exploratorio de Datos (EDA) y Modelado Predictivo para diagnosticar enfermedades cardíacas utilizando 13 variables clínicas. El proyecto está dividido en dos fases principales:

Fase 1: Análisis Exploratorio de Datos (EDA)

Fase 2: Modelado Predictivo

REQUISITOS PREVIOS

Entorno de Ejecución

Google Colab (recomendado) o Jupyter Notebook local

Python 3.8 o superior

Dataset Necesario

Archivo: heart.csv

Descripción: Dataset con 13 variables clínicas y 1 variable objetivo (target)

Disponible en: UCI Machine Learning Repository - Heart Disease

ESTRUCTURA DEL PROYECTO

Fase 1: EDA (Análisis Exploratorio)

Instalación de librerías

Carga y exploración de datos

Análisis estadístico descriptivo

Visualizaciones:

Distribución de variables

Matriz de correlación

Comparación entre grupos

Análisis de outliers

Generación de insights

Fase 2: Modelado Predictivo

Preprocesamiento de datos

División de datos (train/val/test)

Entrenamiento de múltiples modelos:

Regresión Logística

Árbol de Decisión

Random Forest

XGBoost

Evaluación comparativa

Selección del mejor modelo

Análisis de resultados clínicos

GUÍA PASO A PASO

PASO 1: CONFIGURACIÓN INICIAL

En Google Colab:

Abre Google Colab

Crea un nuevo notebook

Copia y pega TODO el código proporcionado (solo una vez, no las repeticiones)

Ejecuta la primera celda de instalación de librerías

Localmente (Jupyter): Instala las dependencias pip install pandas numpy matplotlib seaborn scikit-learn plotly xgboost

PASO 2: CARGA DE DATOS

Opción A: Subir archivo manualmente

Al ejecutar la celda que dice print("Sube el archivo heart.csv")

Se abrirá un botón para seleccionar archivo

Selecciona tu archivo heart.csv

Opción B: Cargar desde URL Alternativa: Descargar dataset automáticamente !wget https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

Renombrar y cargar column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'] df = pd.read_csv('processed.cleveland.data', names=column_names)

PASO 3: EJECUCIÓN DEL EDA

Ejecuta todas las celdas en orden

El código generará automáticamente:

Gráficos de distribución

Matrices de correlación

Análisis estadísticos

Archivo insights_eda.txt con conclusiones

Imágenes de gráficos para reporte

PASO 4: EJECUCIÓN DEL MODELADO

Continúa con la sección de modelado predictivo

El código realizará automáticamente:

Preprocesamiento de datos

Entrenamiento de 4 modelos

Optimización de hiperparámetros

Evaluación comparativa

Guardado del mejor modelo

PASO 5: INTERPRETACIÓN DE RESULTADOS

Archivos generados:

insights_eda.txt: Conclusiones del análisis exploratorio

mejor_modelo_cardio.pkl: Modelo entrenado para predicciones

metricas_prueba.json: Métricas de evaluación

comparativa_modelos.csv: Comparación entre modelos

distribucion_enfermedad.png: Gráfico principal

Métricas clave a observar:

F1-Score: Balance entre precisión y sensibilidad

AUC-ROC: Capacidad discriminativa del modelo

Sensibilidad (Recall): Capacidad para detectar enfermos

Precisión: Confiabilidad de diagnósticos positivos

PERSONALIZACIÓN Y AJUSTES

Modificar parámetros del modelo: En la sección de hiperparámetros, puedes ajustar: param_grids = { 'Random Forest': { 'classifier__n_estimators': [50, 100, 200], Añadir más 'classifier__max_depth': [5, 10, 15, None], 'classifier__min_samples_split': [2, 5, 10] } }

Agregar nuevos modelos: Añadir después de la definición de modelos from sklearn.neighbors import KNeighborsClassifier

models['KNN'] = KNeighborsClassifier() param_grids['KNN'] = { 'classifier__n_neighbors': [3, 5, 7], 'classifier__weights': ['uniform', 'distance'] }

Cambiar división de datos: Modificar proporciones (actual: 70/15/15) X_train, X_temp, y_train, y_temp = train_test_split( X, y, test_size=0.3, stratify=y, random_state=42 ) X_val, X_test, y_val, y_test = train_test_split( X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42 )

ANÁLISIS DE RESULTADOS

Interpretación del mejor modelo:

Variables más importantes: Ver gráfico de importancia

Matriz de confusión: Analizar falsos positivos/negativos

Curva ROC: Evaluar capacidad discriminativa

Ejemplo de predicción con nuevo paciente: Código proporcionado al final del notebook nuevo_paciente = pd.DataFrame({ 'age': [55], 'sex': [1], 'cp': [2], 'trestbps': [130], 'chol': [250], 'fbs': [0], 'restecg': [1], 'thalach': [150], 'exang': [0], 'oldpeak': [1.2], 'slope': [2], 'ca': [0], 'thal': [2] })

pred = modelo.predict(nuevo_paciente)[0] prob = modelo.predict_proba(nuevo_paciente)[0][1]

SOLUCIÓN DE PROBLEMAS COMUNES

Problema 1: Error al subir archivo Solución: Asegúrate que el archivo se llame exactamente heart.csv

Problema 2: Librerías no instaladas Solución: Ejecutar manualmente: !pip install pandas numpy matplotlib seaborn scikit-learn plotly xgboost -q

Problema 3: Memoria insuficiente (Colab) Solución:

Reiniciar entorno de ejecución

Usar solo secciones específicas del código

Reducir tamaño de GridSearchCV

Problema 4: Variables categóricas con valores inesperados Solución: Verificar valores únicos: print(df['thal'].unique()) print(df['ca'].unique())


SOPORTE

Para problemas técnicos:

Revisar mensajes de error detallados

Verificar versiones de librerías

Consultar documentación de scikit-learn

Revisar que todas las celdas se ejecuten en orden

Para dudas académicas:

Revisar significado clínico de variables en el diccionario proporcionado

Consultar bibliografía sobre enfermedad cardíaca

Validar resultados con literatura médica

Ejecuta el código paso a paso y analiza los resultados
