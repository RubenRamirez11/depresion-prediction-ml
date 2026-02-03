# Predicción de Depresión mediante Regresión Logística

## Descripción
Este repositorio contiene un análisis de aprendizaje supervisado cuyo
objetivo es predecir la presencia de síntomas depresivos a partir de
variables psicológicas y sociodemográficas.

Se evaluaron múltiples modelos de clasificación mediante validación
cruzada, seleccionándose finalmente un modelo de regresión logística
por su mejor desempeño y estabilidad.

## Datos
El análisis utiliza un dataset con información sobre características
psicológicas y variables asociadas al bienestar emocional.
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
