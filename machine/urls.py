from django.urls import path
from . import views


urlpatterns = [
    path('<str:path>/', views.machine, name="machine"),
]
