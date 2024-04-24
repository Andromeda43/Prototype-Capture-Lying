import pickle
import warnings

import pandas as pd


class LandmarksProcessor:
    def __init__(self, landmarks_data, face_type):
        self.landmarks_data = landmarks_data
        self.face_type = face_type

    def transform_to_dataframe(self):
        columns = columns = [f'landmark_{i}_{coord}' for i in range(0, 468) for coord in ['x', 'y', 'z']]

        face_type_columns = [f'FaceType_{ft}' for ft in
                             ['alargado', 'circular', 'corazon', 'cuadrado', 'diamante', 'ovalado',
                              'rectangular', 'triangular a', 'triangular v']]
        columns += face_type_columns

        data = []

        for landmarks_data_point in self.landmarks_data:
            row = []

            # Obtener las coordenadas X, Y, Z del landmark 0
            x_pivote = landmarks_data_point['landmarks'][0]
            y_pivote = landmarks_data_point['landmarks'][1]
            z_pivote = landmarks_data_point['landmarks'][2]

            for i in range(0, 468):
                # Ajustar las coordenadas restando las coordenadas del landmark 0
                row.append(landmarks_data_point['landmarks'][i * 3] - x_pivote)
                row.append(landmarks_data_point['landmarks'][i * 3 + 1] - y_pivote)
                row.append(landmarks_data_point['landmarks'][i * 3 + 2] - z_pivote)

            # Agregar 1 si el tipo de rostro coincide, 0 de lo contrario
            for ft in ['rectangular', 'diamante', 'ovalado', 'corazon', 'cuadrado', 'circular', 'alargado', 'triangular v', 'triangular a']:
                row.append(1 if self.face_type == ft else 0)

            data.append(row)

        df = pd.DataFrame(data, columns=columns)
        return df
    def predict_lie(self, df):
        if not df.empty:
            data_to_predict = df

            predict_model = modelKNearest.predict(data_to_predict)


            # Devolver las predicciones
            return predict_model
        else:
            print("no hay datos")
            return None


# Desactivar FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

with open('K-Nearest-Neighbors_model.pickle', 'rb') as f:
    modelKNearest = pickle.load(f)