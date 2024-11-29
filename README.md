# TrabPythonWebMachineLearning


# Aplicação Web de Análise de Dados e Predições com Machine Learning

Este projeto tem como objetivo criar uma aplicação web em Python que permita ao usuário realizar o upload de arquivos de dados no formato .csv e realizar análises visuais interativas, além de gerar predições personalizadas com base em modelos de machine learning. A aplicação oferece diferentes tipos de visualizações e permite a utilização de classificadores configuráveis pelo usuário.

## Funcionalidades

- *Upload de Arquivos CSV*: Permite ao usuário fazer o upload de arquivos de dados no formato .csv para serem analisados.
- *Análises Visuais*: Geração de gráficos interativos para explorar os dados, como gráficos de dispersão, histograma, boxplot, etc.
- *Predições com Machine Learning*: Utilização de classificadores (como Random Forest,  entre outros) para gerar predições com base nos dados carregados.
- *Matrizes de Correlação*: Exibição de correlações entre variáveis para facilitar a análise exploratória.

## Estrutura da Aplicação

A aplicação está dividida em três principais módulos:

1. *Carregamento do Dataset*: Permite o upload de arquivos .csv e a pré-visualização dos dados.
2. *Análises*: O usuário pode visualizar diferentes gráficos interativos para explorar os dados de forma intuitiva.
3. *Predições*: O modelo de machine learning permite ao usuário treinar e realizar previsões sobre os dados carregados.

## Tecnologias Utilizadas

- *Python*: Linguagem principal para a aplicação.
- *Django*: Framework web utilizado para construir a interface da aplicação.
- *Pandas*: Biblioteca para manipulação de dados.
- *Scikit-learn*: Biblioteca de machine learning utilizada para criar os modelos preditivos.
- *Plotly/Matplotlib*: Bibliotecas para visualização interativa e gráfica dos dados.

## Imports

```python
from django.shortcuts import render, redirect
import os
import pandas as pd
import numpy as np
from django.core.files.storage import FileSystemStorage
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import base64
from django.http import JsonResponse
from django.conf import settings
import seaborn as sns
import matplotlib
from pathlib import Path

matplotlib.use('Agg')