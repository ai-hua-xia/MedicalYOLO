import os

from yoloserver.yoloserver.settings import BASE_DIR


INSTALLED_APPS = [
    # ...
    'medical_app',
]

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# 允许上传的文件大小限制
DATA_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024  # 100MB