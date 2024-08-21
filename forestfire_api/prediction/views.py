from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import numpy as np
import pandas as pd

# Create your views here.
class PredictAreaView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = joblib.load('prediction/forestfire_model.pkl')

    def post(self, request):
        data = request.data
        try:
            # Extraer las características de la solicitud
            features = {
                'X': [data['X']],
                'Y': [data['Y']],
                'FFMC': [data['FFMC']],
                'DMC': [data['DMC']],
                'DC': [data['DC']],
                'ISI': [data['ISI']],
                'temp': [data['temp']],
                'RH': [data['RH']],
                'wind': [data['wind']],
                'rain': [data['rain']],
                'month': [data['month']],
                'day': [data['day']]
            }

            # Convertir las características a un DataFrame
            features_df = pd.DataFrame(features)

            # Realizar la predicción
            predicted_area = self.model.predict(features_df)[0]

            return Response({'predicted_area': predicted_area}, status=status.HTTP_200_OK)

        except KeyError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
