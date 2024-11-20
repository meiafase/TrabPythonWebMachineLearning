import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
import os
import matplotlib
matplotlib.use('Agg')  # Configurar backend não-interativo
import matplotlib.pyplot as plt



def home(request):
    return render(request, 'home/Home.html');



def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']

        # Salvar o arquivo em um diretório local (media/uploads)
        fs = FileSystemStorage(location='media/uploads')
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        try:
            # Ler o arquivo com pandas
            cleaned_data_df = pd.read_csv(file_path)

            # Cálculo da contagem média de jogadores para cada plataforma
            windows_player_counts = cleaned_data_df[cleaned_data_df['PlatformWindows'] == True]['SteamSpyOwners'].mean()
            linux_player_counts = cleaned_data_df[cleaned_data_df['PlatformLinux'] == True]['SteamSpyOwners'].mean()
            mac_player_counts = cleaned_data_df[cleaned_data_df['PlatformMac'] == True]['SteamSpyOwners'].mean()

            # Gráfico de barras para visualizar
            platforms = ['Windows', 'Linux', 'Mac']
            average_player_counts = [windows_player_counts, linux_player_counts, mac_player_counts]
            colors = ['blue', 'purple', 'orange']

            plt.bar(platforms, average_player_counts, color=colors)
            plt.title('Média de jogadores por plataforma')
            plt.xlabel('Plataforma')
            plt.ylabel('Contagem média de jogadores (Milhares)')

            # Salvar o gráfico no diretório de uploads
            graph_filename = 'platform_average_players.jpg'
            graph_path = os.path.join(fs.location, graph_filename)
            plt.savefig(graph_path, bbox_inches='tight')
            plt.close()  # Fecha o gráfico para liberar memória

        except Exception as e:
            return render(request, 'home/Result.html', {'error': f'Erro ao processar o arquivo: {str(e)}'})

        # Obter a URL pública do gráfico salvo
        graph_url = fs.url(graph_filename)

        # Renderizar o template com a URL do gráfico
        return render(request, 'home/Result.html', {'graph_url': graph_url})

    return render(request, 'home/Upload.html')