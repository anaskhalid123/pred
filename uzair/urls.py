from django.conf.urls import url
from .views import ap

urlpatterns=[
    url(r'^/project/',ap)
]