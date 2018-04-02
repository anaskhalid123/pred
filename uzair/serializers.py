from rest_framework import serializers
from .models import Tests
class testSerializer(serializers.ModelSerializer):
    class Meta:
        model=Tests
        fields="__all__"