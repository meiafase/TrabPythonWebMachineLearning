from django.shortcuts import render, redirect
import os
import pandas as pd
import numpy as np
from django.core.files.storage import FileSystemStorage
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import base64
from django.http import JsonResponse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from django.conf import settings



def machine(request, path):
    if request.method == 'POST':

        try:
            cleaned_data_df = pd.read_csv(path)

            X_columns = request.POST.getlist('features')
            y_column = request.POST.get('target') 

            X = cleaned_data_df[X_columns]  
            y = cleaned_data_df[y_column]  

            tarefa = request.POST.get('tarefa')
            modelo = request.POST.get('modelo')
            max_depth = int(request.POST.get('max_depth'))
            n_neighbors = int(request.POST.get('n_neighbors'))
            n_estimators = int(request.POST.get('n_estimators'))
            learning_rate = float(request.POST.get('learning_rate')) / 100
            print(learning_rate)

            # Dividir os dados em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if tarefa == "Regressão":
                if modelo == "Gradient Boosting Regression":
                    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                elif modelo == "Random Forest Regression":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif modelo == "Decision Tree Regression":
                    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                elif modelo == "KNN Regression":
                    model = KNeighborsRegressor(n_neighbors=n_neighbors)
                else:
                    return "Modelo não suportado para regressão.", None
            elif tarefa == "Classificação":
                if modelo == "Random Forest Regression":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif modelo == "Decision Tree Regression":
                    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                elif modelo == "KNN Regression":
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                else:
                    return "Modelo não suportado para classificação.", None
            else:
                return "Tarefa não suportada.", None
            
            
            # Treina o modelo
            model.fit(X_train, y_train)

            # Faz previsões
            y_pred = model.predict(X_test)

            # Calcular métricas de avaliação
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            texto_explicativo = f"""
            O modelo selecionado foi o {modelo} para a tarefa de {tarefa}. 
            O target escolhido foi {y_column} e as features selecionadas foram {', '.join(X_columns)}.
            O erro médio quadrático (MSE) foi de {mse:.2f}, o que indica quão bem o modelo se ajusta aos dados. 
            Quanto menor o MSE, melhor o modelo. 
            O coeficiente de determinação R² foi de {r2:.2f} o que significa que {r2*100:.2f}% da variabilidade dos dados pode ser explicada pelo modelo.
            """

            return render(request, 'machine/Machine.html', {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "texto_explicativo": texto_explicativo,
                "path": path
            })
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return render(request, 'machine/Machine.html', {"path": path})
 
