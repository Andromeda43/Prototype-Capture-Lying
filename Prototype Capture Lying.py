import pickle
import pandas as pd
import tkinter as tk
import cv2
import mediapipe as mp
from threading import Thread
from collections import deque
import time
from tkinter import ttk
import warnings


class LyingDetectionInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Lying Detection")

        # Obtener dimensiones de la pantalla
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Calcular posición central
        x_position = (screen_width - 350) // 2  # Cuadro de 350 píxeles de ancho
        y_position = (screen_height - 200) // 2  # Cuadro de 200 píxeles de alto

        # Configurar posición y tamaño de la ventana
        self.root.geometry(f"350x200+{x_position}+{y_position}")

        # Crear etiqueta de título
        title_label = tk.Label(root, text="Lying Detection", font=("Helvetica", 16))
        title_label.grid(row=0, column=0, columnspan=2, pady=(10, 15), padx=10)

        # Checklist para seleccionar el tipo de rostro
        face_type_label = tk.Label(root, text="Seleccionar tipo de rostro:")
        face_type_label.grid(row=1, column=0, sticky=tk.W, pady=5, padx=10)
        self.face_type_var = tk.StringVar(value="rectangular")  # Valor predeterminado
        face_types = ['rectangular', 'cuadrado', 'circular', 'corazon', 'alargado', 'diamante', 'ovalado', 'triangular v', 'triangular a']
        face_type_checklist = ttk.Combobox(root, textvariable=self.face_type_var, values=face_types, state="readonly")
        face_type_checklist.grid(row=1, column=1, pady=5, padx=10)

        # Checklist para seleccionar la cámara
        camera_label = tk.Label(root, text="Seleccionar cámara:")
        camera_label.grid(row=2, column=0, sticky=tk.W, pady=5, padx=10)
        self.camera_var = tk.StringVar(value=self.get_available_cameras()[0])  # Valor predeterminado
        camera_options = self.get_available_cameras()
        camera_checklist = ttk.Combobox(root, textvariable=self.camera_var, values=camera_options, state="readonly")
        camera_checklist.grid(row=2, column=1, pady=5, padx=10)

        # Botón para iniciar la detección
        start_detection_button = tk.Button(root, text="Iniciar Detección", command=self.start_detection)
        start_detection_button.grid(row=3, column=0, columnspan=2, pady=(10, 15), padx=10)

    def get_available_cameras(self):
        # Obtener las cámaras disponibles usando OpenCV
        camera_list = []
        for i in range(10):  # Puedes ajustar el rango según tus necesidades
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_list.append(str(i))
                cap.release()
        return camera_list

    def start_detection(self):
        # Obtener los valores seleccionados
        selected_face_type = self.face_type_var.get()
        selected_camera = self.camera_var.get()

        # Cerrar la interfaz principal antes de abrir la nueva ventana
        self.root.destroy()

        # Crear una nueva instancia de Tkinter para la aplicación de malla facial
        root = tk.Tk()
        face_mesh_app = FaceMeshApp(root, selected_camera, selected_face_type)
        root.mainloop()


class FaceMeshApp:
    def __init__(self, root, camera_index, face_type):

        self.camera_index = int(camera_index)
        self.cap = cv2.VideoCapture(self.camera_index)

        self.face_type = face_type

        # Configuración de MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Crear instancia de FaceLandmarker
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Iniciar hilo para procesar la malla facial en tiempo real
        self.thread = Thread(target=self.process_face_mesh, daemon=True)
        self.thread.start()

        # Almacenar la secuencia de frames
        self.frames_sequence = deque(maxlen=30)  # Ajusta el tamaño de la secuencia según tus necesidades

        # Inicializar el buffer de predicciones
        self.predictions_buffer = []

        # Crear una etiqueta para mostrar el porcentaje de predicciones
        self.prediction_label_var = tk.StringVar()
        self.prediction_label = tk.Label(root, textvariable=self.prediction_label_var)
        self.prediction_label.pack()

    def process_face_mesh(self):
        predictions_buffer = []  # Lista para almacenar las predicciones durante el intervalo
        interval_start_time = time.time()

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Convertir la imagen a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar la imagen con MediaPipe Face Mesh
            landmarks_results = self.face_mesh.process(rgb_frame)

            # Dibujar los landmarks en la imagen
            if landmarks_results.multi_face_landmarks:
                landmarks_data = {'landmarks': []}

                for landmarks in landmarks_results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                                                   landmark_drawing_spec=self.mp_drawing.DrawingSpec(thickness=1,
                                                                                                     circle_radius=1))

                    # Pasar los datos de los landmark
                    for landmark in landmarks.landmark:
                        landmarks_data['landmarks'].extend([landmark.x, landmark.y, landmark.z])

                # Obtener el ancho y alto del marco
                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Definir el nuevo tamaño de la ventana
                new_width = 720
                new_height = 480

                # Redimensionar el marco al nuevo tamaño
                frame = cv2.resize(frame, (new_width, new_height))
                # Mostrar la imagen en la ventana con OpenCV
                cv2.imshow('Face Mesh Landmarks', frame)

                # Manejar los landmarks
                self.handle_landmarks(landmarks_data, predictions_buffer)

                # Calcular el tiempo transcurrido
                current_time = time.time()
                elapsed_time = current_time - interval_start_time

                valueTrue = "100"

                # Realizar predicciones durante el intervalo de 2 segundos
                if elapsed_time >= 1.0:
                    interval_start_time = current_time
                    # Procesar los resultados de las predicciones durante el intervalo
                    self.process_predictions(predictions_buffer)
                    # Limpiar el buffer de predicciones
                    predictions_buffer = []
                # Salir del bucle si se presiona 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    def handle_landmarks(self, landmarks_data, predictions_buffer):
        # Verificar si hay suficientes frames en la secuencia para realizar predicciones
        if len(self.frames_sequence) >= 30:
            # Obtener los valores seleccionados
            selected_face_type = self.face_type

            # Crear instancia de LandmarksProcessor
            landmarks_processor = LandmarksProcessor(list(self.frames_sequence), selected_face_type)
            df = landmarks_processor.transform_to_dataframe()

            # Realizar predicciones
            predictions = landmarks_processor.predict_lie(df)
            #print("Predictions:", predictions)

            # Agregar las predicciones al buffer
            predictions_buffer.extend(predictions)

        # Limpiar all_landmarks para evitar duplicados

        # Agregar el actual frame a la secuencia
        self.frames_sequence.append(landmarks_data)

    def update_prediction_label(self, true_percentage):
        # Actualizar el valor de la etiqueta con el porcentaje de predicciones verdaderas
        self.prediction_label_var.set(f"Porcentaje de Verdad - {true_percentage:.2f}%")

    def process_predictions(self, predictions_buffer):
        # Procesar y mostrar el porcentaje de predicciones verdaderas durante el intervalo
        if predictions_buffer:
            true_count = sum(predictions_buffer)
            total_predictions = len(predictions_buffer)
            true_percentage = (true_count / total_predictions) * 100
            self.update_prediction_label(true_percentage)



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



with open('K-Nearest-Neighbors_model.pickle', 'rb') as f:
    modelKNearest = pickle.load(f)
# Desactivar FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    root = tk.Tk()
    lying_interface = LyingDetectionInterface(root)
    root.mainloop()