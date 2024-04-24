import cv2
from FaceMeshApp import FaceMeshApp
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

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
        start_detection_button.grid(row=3, column=0, columnspan=2, pady=(10, 5), padx=10, sticky=tk.W+tk.E)

        # Botón para mostrar la guía de uso
        info_button = tk.Button(root, text="Guía de Uso", command=self.show_info)
        info_button.grid(row=4, column=0, columnspan=2, pady=(5, 10), padx=10, sticky=tk.W+tk.E)

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

    def show_info(self):
        # Mostrar las instrucciones de uso en un mensaje emergente
        instructions = (
            "Instrucciones de Uso\n\n"
            "1. Ejecución del Proyecto: Inicie el proyecto Prototype-Capture-Lying.\n\n"
            "2. Identificación del Tipo de Rostro: Una vez que el proyecto esté en ejecución, "
            "identifique el tipo de rostro de la persona a entrevistar. Esto implica seleccionar "
            "una opción específica dentro de una casilla desplegable dentro de la interfaz de usuario.\n\n"
            "3. Selección de la Cámara: Seleccione la cámara de su dispositivo que registrará al entrevistador. "
            "Esto requiere seleccionar una opción específica dentro de la interfaz de usuario que indique "
            "las cámaras disponibles detectadas en su dispositivo.\n\n"
            "4. Inicio de la Detección: Presione el botón Iniciar Detección para iniciar el proceso de detección "
            "de gestos faciales y análisis de veracidad.\n\n"
            "5. Posicionamiento del Entrevistador: Una vez que se abra la cámara y aparezca el cuadro de diálogo "
            "que muestra el porcentaje de verdad, ubique al entrevistador a una distancia entre 30 y 40 centímetros "
            "de la cámara. Sabrá que está en la posición correcta cuando el indicador de verdad alcance un porcentaje "
            "del 100%.\n\n"
            "6. Verificación del Contador: Verifique que el contador de porcentaje se mantenga en el 100% durante al menos "
            "4 segundos. Esto garantizará una detección precisa y estable de los gestos faciales del entrevistado.\n\n"
            "7. Realización de Preguntas: Una vez que el contador se haya mantenido estable, proceda con las preguntas de la "
            "entrevista. Visualice los porcentajes de veracidad suministrados por el sistema en tiempo real y utilícelos "
            "como referencia durante la entrevista.\n\n"
            "8. Finalización de la Aplicación: Cuando haya completado la entrevista o el uso de la aplicación, asegúrese de cerrarla correctamente.\n\n"
            "¡Ahora está listo para utilizar la aplicación Prototype-Capture-Lying de manera efectiva!"
        )

        messagebox.showinfo("Guía de Uso", instructions)