# NLP Detección de Sarcasmo

Este repositorio contiene la solución al **Caso de Estudio 3 - Detección de Sarcasmo**, enfocado en el análisis de datos textuales para identificar tonos sarcásticos en locuciones. A continuación, se describen las tareas realizadas y cómo reproducir los resultados.

---

## **Descripción del Problema**
Los archivos `Sarcasmo_train.csv` y `Sarcasmo_test.csv` contienen locuciones marcadas con o sin tono sarcástico. El objetivo principal es:

1. **Procesamiento de Datos:** 
   - Depurar y mejorar la calidad de los datos textuales.
2. **Comparación de Modelos:** 
   - Entrenar y evaluar 4 modelos:
     - **BERT** con y sin ingeniería de variables.
     - **FastText** con y sin ingeniería de variables.
   - Comparar los modelos usando el **área bajo la curva ROC (ROC AUC)** como métrica principal.
3. **Identificación del Mejor Modelo:**
   - Seleccionar el modelo con el mayor valor de ROC AUC al predecir sobre los datos de prueba.
4. **Pipeline y Automatización:**
   - Desarrollar un pipeline que automatice el procesamiento, entrenamiento y evaluación.
   - Implementar pruebas unitarias y de integración que validen:
     - La correcta selección del mejor modelo.
     - La generación de un archivo CSV con las métricas del conjunto de prueba para cada modelo.

---

## **Ejecución del Código**

Main:
`python -m src.main`

Pruebas de integración:
`python -m unittest discover -s tests`
