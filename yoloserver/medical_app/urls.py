from django.urls import path
from .views import upload_file, infer_and_display

urlpatterns = [
    path('upload/', upload_file, name='upload_file'),
    path('infer/', infer_and_display, name='infer_and_display'),
]