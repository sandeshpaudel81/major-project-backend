from django.apps import AppConfig
from pathlib import Path
from ultralytics import YOLO

BASE_DIR_MODEL = Path(__file__).resolve().parent

class AppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "app"

    def ready(self):
        frontPath = BASE_DIR_MODEL / "models/front.pt"
        backPath = BASE_DIR_MODEL / "models/back.pt"
        self.front_model = YOLO(frontPath)
        self.back_model = YOLO(backPath)
