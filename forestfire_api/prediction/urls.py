from django.urls import path
from .views import PredictAreaView

urlpatterns = [
    path('predict/', PredictAreaView.as_view(), name='predict_area'),
]
