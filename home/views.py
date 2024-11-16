from django.shortcuts import render
import csv
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage


def home(request):
    return render(request, 'home/Home.html');


def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        
        # Salvar o arquivo em um diretório local (media/uploads)
        fs = FileSystemStorage(location='media/uploads')
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)
        return render(request, 'home/Result.html')
