    # flake8: noqa: E501
    #
    # En este dataset se desea pronosticar el default (pago) del cliente el próximo
    # mes a partir de 23 variables explicativas.
    #
    #   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
    #              credito familiar (suplementario).
    #         SEX: Genero (1=male; 2=female).
    #   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
    #    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
    #         AGE: Edad (years).
    #       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
    #       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
    #       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
    #       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
    #       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
    #       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
    #   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
    #   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
    #   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
    #   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
    #   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
    #   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
    #    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
    #    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
    #    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
    #    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
    #    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
    #    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
    #
    # La variable "default payment next month" corresponde a la variable objetivo.
    #
    # El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
    # en la carpeta "files/input/".
    #
    # Los pasos que debe seguir para la construcción de un modelo de
    # clasificación están descritos a continuación.
    #
    #
    # Paso 1.
    # Realice la limpieza de los datasets:
    # - Renombre la columna "default payment next month" a "default".
    # - Remueva la columna "ID".
    # - Elimine los registros con informacion no disponible.
    # - Para la columna EDUCATION, valores > 4 indican niveles superiores
    #
    #
    #
    # Paso 2.
    # Divida los datasets en x_train, y_train, x_test, y_test.
    #
    #
    # Paso 3.
    # Cree un pipeline para el modelo de clasificación. Este pipeline debe
    # contener las siguientes capas:
    # - Transforma las variables categoricas usando el método
    #   one-hot-encoding.
    # - Ajusta un modelo de bosques aleatorios (rando forest).
    #
    #
    # Paso 4.
    # Optimice los hiperparametros del pipeline usando validación cruzada.
    # Use 10 splits para la validación cruzada. Use la función de precision
    # balanceada para medir la precisión del modelo.
    #
    #
    # Paso 5.
    # Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
    # Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
    #
    #
    # Paso 6.
    # Calcule las metricas de precision, precision balanceada, recall,
    # y f1-score para los conjuntos de entrenamiento y prueba.
    # Guardelas en el archivo files/output/metrics.json. Cada fila
    # del archivo es un diccionario con las metricas de un modelo.
    # Este diccionario tiene un campo para indicar si es el conjunto
    # de entrenamiento o prueba. Por ejemplo:
    #
    # {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
    # {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
    #
    #
    # Paso 7.
    # Calcule las matrices de confusion para los conjuntos de entrenamiento y
    # prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
    # del archivo es un diccionario con las metricas de un modelo.
    # de entrenamiento o prueba. Por ejemplo:
    #
    # {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
    # {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
    #
    ############################################################################################
    ## Solution

import pandas as pd
import numpy as np
import os
import gzip
import pickle
import json
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    confusion_matrix
)

# ============================
# Paso 1. Cargar y limpiar datos
# ============================
def load_data():
    train_path = 'files/input/train_data.csv.zip'
    test_path = 'files/input/test_data.csv.zip'

    train_dataset = pd.read_csv(train_path, index_col=False)
    test_dataset = pd.read_csv(test_path, index_col=False)
    return train_dataset, test_dataset


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={"default payment next month": "default"}, inplace=True)

    df.drop(columns=["ID"], inplace=True)

    df = df.loc[df["MARRIAGE"] != 0]
    df = df.loc[df["EDUCATION"] != 0]

    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    df.dropna(inplace=True)

    # Eliminar duplicados
    df = df.drop_duplicates()

    return df

# =====================
# Paso 2: Separar variables y objetivo
# =====================


def split_features_target(train_dataset, test_dataset, target_col="default"):
    x_train = train_dataset.drop(columns=[target_col])
    y_train = train_dataset[target_col].astype(int).values.ravel()
    x_test = test_dataset.drop(columns=[target_col])
    y_test = test_dataset[target_col].astype(int).values.ravel()
    return x_train, y_train, x_test, y_test

def make_train_test_split(x, y):

    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42
    )
    print("Tamaños:")
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    return x_train, x_test, y_train, y_test


# =====================
# Paso 3: Crear pipeline
# =====================


def make_pipeline(list_categorical, estimator):

    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), list_categorical)
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator)
        ]
    )

    return pipeline



# =====================
# Paso 4: Optimización de hiperparámetros
# =====================


def make_grid_search(pipeline, param_grid, cv, score, x_train, y_train):

    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=score,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(x_train, np.ravel(y_train))

    return grid_search


# =====================
# Paso 5: Guardar el modelo
# =====================

def save_estimator(estimator):
    models_path = "files/models"
    os.makedirs(models_path, exist_ok=True)

    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(estimator, file)     
    print(f"Modelo guardado en: {'files/models/model.pkl.gz'}")


def load_estimator(output_path):
    """Cargar modelo comprimido"""
    import gzip, pickle
    if not os.path.exists(output_path):
        return None
    with gzip.open(output_path, "rb") as f:
        return pickle.load(f)


# =====================
# Paso 6: Calcular métricas y guardarlas en un archivo JSON
# =====================

# Calcular métricas para el conjunto de entrenamiento y prueba

def calculate_metrics(model, x_train, y_train, x_test, y_test):

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    metrics = [
        {
            'type': 'metrics',
            'dataset': 'train',
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
        },
        {
            'type': 'metrics',
            'dataset': 'test',
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
        },
        {
            'type': 'cm_matrix',
            'dataset': 'train',
            'true_0': {'predicted_0': int(cm_train[0, 0]), 'predicted_1': int(cm_train[0, 1])},
            'true_1': {'predicted_0': int(cm_train[1, 0]), 'predicted_1': int(cm_train[1, 1])}
        },
        {
            'type': 'cm_matrix',
            'dataset': 'test',
            'true_0': {'predicted_0': int(cm_test[0, 0]), 'predicted_1': int(cm_test[0, 1])},
            'true_1': {'predicted_0': int(cm_test[1, 0]), 'predicted_1': int(cm_test[1, 1])}
        }
    ]

    return metrics

def save_metrics(metrics):
    metrics_path = "files/output"
    os.makedirs(metrics_path, exist_ok=True)
    
    with open("files/output/metrics.json", "w") as file:
        for metric in metrics:
            file.write(json.dumps(metric, ensure_ascii=False))
            file.write('\n')
    
    with open("files/output/metrics.json", "r") as file:
        print("Contenido de metrics.json:")
        print(file.read())

########################################################################
### Orquestador entrenanmiento y evaluación del modelo
########################################################################
def main():

    train_dataset, test_dataset = load_data()

    train_dataset = clean_dataset(train_dataset)
    test_dataset = clean_dataset(test_dataset)

    x_train, y_train, x_test, y_test = split_features_target(train_dataset, test_dataset, "default")

    list_categorical = ["EDUCATION", "MARRIAGE", "SEX"]

    pipeline = make_pipeline(list_categorical, RandomForestClassifier(random_state=42))

    # Definir los hiperparámetros a optimizar
    param_grid = {
        "classifier__n_estimators": [30, 50, 100],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [5, 10],
        "classifier__min_samples_leaf": [2, 4],
        "classifier__max_features": [10, 25]
    }


    # Ajustar el modelo con los datos de entrenamiento
    # En estimator se pasa el pipeline
    model = make_grid_search(pipeline, param_grid, cv=10, score="balanced_accuracy", x_train=x_train, y_train=y_train)

    save_estimator(model)

    metrics = calculate_metrics(model, x_train, y_train, x_test, y_test)

    save_metrics(metrics)

if __name__ == "__main__":
    main()

