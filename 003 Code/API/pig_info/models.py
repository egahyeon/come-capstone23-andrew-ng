from django.db import models

# Create your models here.
class pig_info(models.Model):
    pNo = models.IntegerField()
    now = models.DateTimeField()
    act = models.FloatField() 
    pred = models.BooleanField()
