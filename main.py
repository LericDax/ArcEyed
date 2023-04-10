import sys
import cv2
import math
import dlib
import numpy as np
import glob
import os
import torch
import openai
import bz2
import clip
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, QVBoxLayout, QWidget

class EyebrowMatcher:

    @staticmethod
    def load_dlib_models():
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        face_recognition_model_path = "dlib_face_recognition_resnet_model_v1.dat"

        predictor = dlib.shape_predictor(predictor_path)
        face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

        return predictor, face_recognition_model
        
    @staticmethod
    def detect_eyebrows(image, predictor):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(gray_image)

        eyebrows = []
        for face in faces:
            landmarks = predictor(gray_image, face)
            left_eyebrow = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)])
            right_eyebrow = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)])
            eyebrows.append((left_eyebrow, right_eyebrow))

        return eyebrows

    @staticmethod
    def extract_eyebrow_region(image, eyebrows):
        left_eyebrow, right_eyebrow = eyebrows
        x_min = min(np.min(left_eyebrow[:, 0]), np.min(right_eyebrow[:, 0]))
        x_max = max(np.max(left_eyebrow[:, 0]), np.max(right_eyebrow[:, 0]))
        y_min = min(np.min(left_eyebrow[:, 1]), np.min(right_eyebrow[:, 1]))
        y_max = max(np.max(left_eyebrow[:, 1]), np.max(right_eyebrow[:, 1]))

        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)

        return image[y_min:y_max, x_min:x_max]

    @staticmethod
    def preprocess_image(image, target_size=(224, 224)):
        image = cv2.resize(image, target_size)
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return image


    @staticmethod
    def match_eyebrows(source_image_path, image_folder_path, output_folder_path, threshold=0.5):
        predictor, face_recognition_model = EyebrowMatcher.load_dlib_models()

        source_image = cv2.imread(source_image_path)
        source_eyebrows = EyebrowMatcher.detect_eyebrows(source_image, predictor)
        if not source_eyebrows:
            return []

        source_face_descriptor = EyebrowMatcher.get_face_descriptor(source_image, predictor, face_recognition_model)
        # Convert source_face_descriptor to a NumPy array
        source_face_descriptor = np.array(source_face_descriptor)

        image_files = glob.glob(image_folder_path + '/*')
        image_files = [img for img in image_files if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        matched_images = []

        for image_path in image_files:
            image = cv2.imread(image_path)
            eyebrows = EyebrowMatcher.detect_eyebrows(image, predictor)

            if eyebrows:
                face_descriptor = EyebrowMatcher.get_face_descriptor(image, predictor, face_recognition_model)
                if face_descriptor is not None:
                    # Convert face_descriptor to a NumPy array
                    face_descriptor = np.array(face_descriptor)
                    # Calculate Euclidean distance between face descriptors
                    distance = np.linalg.norm(source_face_descriptor - face_descriptor)
                    similarity = 1 - distance
                    if similarity > threshold:
                        matched_images.append(image_path)
                        output_path = os.path.join(output_folder_path, os.path.basename(image_path))
                        cv2.imwrite(output_path, image)

        return matched_images


        
    @staticmethod
    def get_face_descriptor(image, predictor, face_recognition_model):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use the face detector to get face detections
        face_detector = dlib.get_frontal_face_detector()
        face_detections = face_detector(gray_image)
        
        # If no face is detected, return None
        if not face_detections:
            return None
        
        # Use the predictor to get facial landmarks for the first face detection
        landmarks = predictor(gray_image, face_detections[0])
        
        # Call dlib.get_face_chip_details with the dlib.full_object_detection object
        rectangle = dlib.get_face_chip_details(landmarks, size=150, padding=0.25)
        
        # Convert dlib.drectangle to dlib.rectangle
        rect = dlib.rectangle(int(rectangle.rect.left()), int(rectangle.rect.top()),
                              int(rectangle.rect.right()), int(rectangle.rect.bottom()))
        
        shape = predictor(gray_image, rect)

        # Convert the grayscale image back to a 3-channel color image
        color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        face_descriptor = face_recognition_model.compute_face_descriptor(color_image, shape)
        return np.array(face_descriptor)








class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Eyebrow Matcher")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.source_image_label = QLabel("Source Image Path:")
        self.source_image_edit = QLineEdit()
        self.source_image_button = QPushButton("Browse")
        self.source_image_button.clicked.connect(self.browse_source_image)

        self.layout.addWidget(self.source_image_label)
        self.layout.addWidget(self.source_image_edit)
        self.layout.addWidget(self.source_image_button)

        self.image_folder_label = QLabel("Image Folder Path:")
        self.image_folder_edit = QLineEdit()
        self.image_folder_button = QPushButton("Browse")      
        self.image_folder_button.clicked.connect(self.browse_image_folder)

        self.layout.addWidget(self.image_folder_label)
        self.layout.addWidget(self.image_folder_edit)
        self.layout.addWidget(self.image_folder_button)

        self.output_folder_label = QLabel("Output Folder Path:")
        self.output_folder_edit = QLineEdit()
        self.output_folder_button = QPushButton("Browse")
        self.output_folder_button.clicked.connect(self.browse_output_folder)

        self.layout.addWidget(self.output_folder_label)
        self.layout.addWidget(self.output_folder_edit)
        self.layout.addWidget(self.output_folder_button)

        self.match_button = QPushButton("Match Eyebrows")
        self.match_button.clicked.connect(self.match_eyebrows)

        self.layout.addWidget(self.match_button)

        self.results_label = QLabel("Matched Images:")
        self.results_edit = QTextEdit()

        self.layout.addWidget(self.results_label)
        self.layout.addWidget(self.results_edit)

    def browse_source_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Source Image", "", "Images (*.jpg *.jpeg *.png)", options=options)
        if file_name:
            self.source_image_edit.setText(file_name)

    def browse_image_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        folder_name = QFileDialog.getExistingDirectory(self, "Select Image Folder", options=options)
        if folder_name:
            self.image_folder_edit.setText(folder_name)

    def browse_output_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        folder_name = QFileDialog.getExistingDirectory(self, "Select Output Folder", options=options)
        if folder_name:
            self.output_folder_edit.setText(folder_name)

    def match_eyebrows(self):
        source_image_path = self.source_image_edit.text()
        image_folder_path = self.image_folder_edit.text()
        output_folder_path = self.output_folder_edit.text()
        matched_images = EyebrowMatcher.match_eyebrows(source_image_path, image_folder_path, output_folder_path)

        self.results_edit.clear()
        for matched_image in matched_images:
            self.results_edit.append(matched_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

