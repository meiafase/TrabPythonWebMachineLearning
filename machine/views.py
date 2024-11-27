from django.shortcuts import render, redirect
import os
import pandas as pd
import numpy as np
from django.core.files.storage import FileSystemStorage
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
from sklearn.metrics import mean_squared_error
from django.conf import settings


def generate_scatter_plot(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.set_title('Gráfico Scatter de predição')

    # Cria o diretório "plots" dentro da pasta "media", caso não exista
    plot_dir = os.path.join(settings.BASE_DIR, 'media')
    os.makedirs(plot_dir, exist_ok=True)

    # Define o caminho do arquivo onde a imagem será salva
    plot_path = os.path.join(plot_dir, 'scatter_plot.png')

    # Salva o gráfico no arquivo
    fig.savefig(plot_path)
    plt.close(fig)

    # Retorna o caminho da imagem salva
    return plot_path

def machine(request):
    if request.method == 'POST':

        try:
            # Recebe os dados do frontend
            X = ""
            y = ["", "", ""]
            tarefa = request.POST.get('tarefa')
            modelo = request.POST.get('modelo')
            max_depth = int(request.POST.get('max_depth'))
            n_neighbors = int(request.POST.get('n_neighbors'))
            n_estimators = int(request.POST.get('n_estimators'))
            learning_rate = float(request.POST.get('learning_rate'))

            # Gerando dados fictícios para treino
            X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

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

            # Calcular o erro médio quadrático (MSE) para avaliação
            mse = mean_squared_error(y_test, y_pred)

            # Gerar gráfico de dispersão das previsões vs valores reais
            img_str = generate_scatter_plot(y_test, y_pred)

            # Retorna a resposta com o MSE e a imagem do gráfico
            return redirect('http://127.0.0.1:8000/machine/')
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        fs = FileSystemStorage(location='media')

        graph_filename_three = 'scatter_plot.png'
        graph_path_three = os.path.join(fs.location, graph_filename_three)
        scatter_plot = fs.url(graph_filename_three)

        return render(request, 'machine/Machine.html', { "scatter_plot_url": scatter_plot })
    
 
