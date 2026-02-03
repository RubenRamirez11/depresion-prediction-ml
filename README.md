# Comparison and Selection of Supervised Learning Models for the Prediction of Depression in University Students

## Description
This repository contains a supervised learning analysis aimed at
predicting the presence of depressive symptoms based on psychological
and sociodemographic variables.

Multiple classification models were evaluated using cross-validation,
with ROC-AUC as the performance metric. Based on this criterion, logistic
regression was identified as the model with the best overall performance.

## Data
The analysis uses a publicly available dataset obtained from Kaggle:
https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset/data

The dataset is provided in CSV format.

## Methodology
- Split of the data into training and test sets
- Cross-validation for model comparison
- Selection of the model with the best predictive performance
- Final training of the selected model on the full training set
- Final evaluation on the test set using ROC curves

## Final Model
The final model corresponds to a binary logistic regression,
trained using all observations from the training set and subsequently
evaluated on the test set to estimate its generalization performance.

## Reproducibility
To reproduce the analysis, run the `modelo.R` script from the root
directory of the repository.


------------------------------------------------------------------------------------------------------------------------------


# Comparación y selección de modelos de aprendizaje supervisado para la predicción de Depresión en estudiantes universitarios

## Descripción
Este repositorio contiene un análisis de aprendizaje supervisado cuyo
objetivo es predecir la presencia de síntomas depresivos a partir de
variables psicológicas y sociodemográficas.

Se evaluaron múltiples modelos de clasificación mediante validación
cruzada, usando como medida de rendimiento el AUC-ROC. En base a esta métrica,
se encontró finalmente que el modelo de regresión logística es el que mejor
desempeño evidencia.

## Datos
Para el análisis se hizo uso de la base de datos de acceso público en Kaggle:
https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset/data

El archivo de datos se encuentra en formato CSV.

## Metodología
- Separación de los datos en conjunto de entrenamiento y prueba
- Validación cruzada para comparación de modelos
- Selección del modelo con mejor desempeño predictivo
- Entrenamiento final del modelo seleccionado sobre el conjunto de entrenamiento
- Evaluación final sobre el conjunto de prueba mediante curvas ROC

## Modelo Final
El modelo final corresponde a una regresión logística binaria,
entrenada utilizando todas las observaciones del conjunto de entrenamiento
y evaluada posteriormente en el conjunto de prueba para estimar su
capacidad de generalización.

## Reproducibilidad
Para reproducir el análisis, ejecutar el archivo `modelo.R`
desde la carpeta raíz del repositorio.
