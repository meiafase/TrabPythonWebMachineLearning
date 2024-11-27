from django.shortcuts import render

def predicoes(request):
    return render(request, 'predicoes/Predicoes.html')
    
 