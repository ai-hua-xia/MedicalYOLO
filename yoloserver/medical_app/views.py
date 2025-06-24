import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO
from pathlib import Path
import subprocess

def upload_file(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'upload_success.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'upload.html')

def infer_and_display(request):
    base_path = Path(settings.BASE_DIR).parent
    model = YOLO(str(base_path / 'models/checkpoints/trainN-20250614_200001-yolov8n-best.pt'))
    source = str(base_path / 'media')
    results = model(source, save=True, save_txt=True)
    result_images = []
    for result in results:
        result_images.append(result.save_dir)
    return render(request, 'infer_results.html', {
        'result_images': result_images
    })