from django.db import models

# Create your models here.
class ImageModel(models.Model):
    front = models.ImageField(upload_to='front-image/', null=True, blank=True)
    back = models.ImageField(upload_to='back-image/', null=True, blank=True)