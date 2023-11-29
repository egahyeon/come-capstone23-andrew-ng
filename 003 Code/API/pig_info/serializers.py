from rest_framework import serializers
from .models import pig_info

class PigInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = pig_info
        fields = ['pNo', 'now', 'act', 'pred']
        
class MainPageSerializer(serializers.Serializer):
    pNo = serializers.IntegerField()
    now = serializers.DateTimeField(allow_null=True, required=False)
    pred = serializers.BooleanField()
        
class GraphPageSerializer(serializers.ModelSerializer):
    class Meta:
        model = pig_info
        fields = ['now', 'act', 'pred']
    # now = serializers.DateTimeField(allow_null=True, required=False)
    # act = serializers.FloatField()
    # pred = serializers.BooleanField()