import time
import tkinter as tk
from collections import deque
import mediapipe as mp
from threading import Thread
import cv2

from LandmarksProcessor import LandmarksProcessor


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