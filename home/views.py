import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
import seaborn as sns
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def home(request):
    return render(request, 'home/Home.html')

def upload(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return render(request, 'home/Upload.html', {'error': 'Nenhum arquivo foi enviado.'})

        uploaded_file = request.FILES['file']
        fs = FileSystemStorage(location='media')
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        try:
            cleaned_data_df = pd.read_csv(file_path)

            required_columns = ['PlatformWindows', 'PlatformLinux', 'PlatformMac', 'SteamSpyOwners', 'GenreIsEarlyAccess']
            if not all(col in cleaned_data_df.columns for col in required_columns):
                return render(request, 'home/Result.html', {'error': 'O arquivo não contém as colunas necessárias.'})

            graph_urls = []

            windows_player_counts = cleaned_data_df[cleaned_data_df['PlatformWindows'] == True]['SteamSpyOwners'].mean()
            linux_player_counts = cleaned_data_df[cleaned_data_df['PlatformLinux'] == True]['SteamSpyOwners'].mean()
            mac_player_counts = cleaned_data_df[cleaned_data_df['PlatformMac'] == True]['SteamSpyOwners'].mean()

            platforms = ['Windows', 'Linux', 'Mac']
            average_player_counts = [windows_player_counts, linux_player_counts, mac_player_counts]
            colors = ['blue', 'purple', 'orange']

            plt.bar(platforms, average_player_counts, color=colors)
            plt.title('Média de jogadores por plataforma')
            plt.xlabel('Plataforma')
            plt.ylabel('Contagem média de jogadores (Milhares)')

            graph_filename_one = 'platform_average_players.jpg'
            graph_path_one = os.path.join(fs.location, graph_filename_one)
            plt.savefig(graph_path_one, bbox_inches='tight')
            plt.close()
            graph_urls.append(fs.url(graph_filename_one))

            player_counts = [
                cleaned_data_df[cleaned_data_df['PlatformWindows']]['SteamSpyOwners'],
                cleaned_data_df[cleaned_data_df['PlatformMac']]['SteamSpyOwners'],
                cleaned_data_df[cleaned_data_df['PlatformLinux']]['SteamSpyOwners']
            ]

            platforms = ['Windows', 'Mac', 'Linux']

            plt.figure(figsize=(10, 6))
            plt.boxplot(player_counts, labels=platforms, showfliers=False)
            plt.title('Distribuição de contagens de jogadores por plataforma (outliers removidos)')
            plt.xlabel('Plataforma')
            plt.ylabel('Contagem de Jogadores (Milhares)')

            graph_filename_two = 'contagem_jogador.jpg'
            graph_path_two = os.path.join(fs.location, graph_filename_two)
            plt.savefig(graph_path_two, bbox_inches='tight')
            plt.close()
            graph_urls.append(fs.url(graph_filename_two))

            mean_player_counts = cleaned_data_df.groupby(
                ['PlatformWindows', 'PlatformMac', 'PlatformLinux']
            )['SteamSpyOwners'].mean()

            num_games_available = cleaned_data_df.groupby(
                ['PlatformWindows', 'PlatformMac', 'PlatformLinux']
            ).size()

            correlation_df = pd.DataFrame({
                'MeanPlayerCounts': mean_player_counts,
                'NumGamesAvailable': num_games_available
            })

            correlation = correlation_df['MeanPlayerCounts'].corr(correlation_df['NumGamesAvailable'])

            plt.figure(figsize=(8, 6))
            for platform, data in correlation_df.groupby(level=[0, 1, 2]):
                if platform == (True, False, False):
                    color = 'blue'
                    platform_label = 'Windows'
                elif platform == (False, True, False):
                    color = 'orange'
                    platform_label = 'Mac'
                elif platform == (False, False, True):
                    color = 'purple'
                    platform_label = 'Linux'

                plt.scatter(
                    data['NumGamesAvailable'],
                    data['MeanPlayerCounts'],
                    label=f'Platform: {platform_label}',
                    c=color
                )

            plt.title(f'Correlação entre número de jogos e média de jogadores (Correlação: {correlation:.2f})')
            plt.xlabel('Número de jogos disponíveis')
            plt.ylabel('Média de jogadores (milhares)')
            plt.legend(title='Plataformas')

            graph_filename_three = 'correlation_games_players.jpg'
            graph_path_three = os.path.join(fs.location, graph_filename_three)
            plt.savefig(graph_path_three, bbox_inches='tight')
            plt.close()
            graph_urls.append(fs.url(graph_filename_three))

            return render(request, 'home/Result.html', {
                'graph_urls': graph_urls
            })

        except Exception as e:
            return render(request, 'home/Result.html', {'error': f'Erro ao processar o arquivo: {str(e)}'})

    return render(request, 'home/Upload.html')
