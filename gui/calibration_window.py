import sys
import time
import cv2
import pyautogui
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent, QGuiApplication
from PyQt5.QtCore import QThreadPool
from .worker import Worker
from backend.utils.landmark import extract_landmark_features, get_model

class CalibrationWindow(QWidget):
    def __init__(self, n_rows: int, n_cols: int, time_delay: float = 1):
        super().__init__()
        QThreadPool.globalInstance().setMaxThreadCount(8)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.time_delay = time_delay
        self.is_running = False
        self._setup_ui()

    def _setup_ui(self):
        self.grid_layout = QGridLayout()

        self.grid_layout.setSpacing(0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                label = QLabel()
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet('border: 1px solid black;')
                self.grid_layout.addWidget(label, i, j)


        self.setLayout(self.grid_layout)
        self.showFullScreen()
        # self.show()

    def get_screen_resolution(self):
        # Get the primary screen resolution
        screen = QApplication.primaryScreen()
        size = screen.size()  # Get the size of the screen
        return size.width(), size.height()  # Return width and height


    def iterateGridSnakes(self):
        # print(QGuiApplication.primaryScreen().availableGeometry())
        # print(QGuiApplication.primaryScreen().geometry())
        # print(QGuiApplication.primaryScreen().virtualGeometry())
        # print(QGuiApplication.primaryScreen().virtualSize())
        self.is_running = True
        cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            raise Exception('Error initializing camera')
            
        for _ in range(10): # discard several frames to allow camera to adjust to lighting
            success, image_raw = cap.read()

        landmarker = get_model('face_landmarker.task')

        def extract_one():
            success, image_raw = cap.read()
            if not success:
                raise Exception('Error reading image from camera')
            image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            face_landmarks, annotated_image = extract_landmark_features(landmarker, image)
            return face_landmarks, annotated_image

        idOdd = True
        for i in range(self.n_rows):
            if idOdd:
                start_idx = 0
                end_idx = self.n_cols
            else:
                start_idx = self.n_cols - 1
                end_idx = -1
            
            for j in range(start_idx, end_idx, 1 if idOdd else -1):
                label = self.grid_layout.itemAtPosition(i, j).widget()
                center_coordinate = label.mapToGlobal(label.rect().center())
                pyautogui.moveTo(center_coordinate.x(), center_coordinate.y())
                label.setStyleSheet('border: 1px solid black; background-color: red;')
                
                time.sleep(0.1)

                label.setStyleSheet('border: 1px solid black; background-color: gray;')

            idOdd = not idOdd
        
        
        cap.release()
        self.is_running = False
        return 'a', 'b'
        
    def _on_iterate_complete(self, x1: str, x2: str):
        pass

    def keyPressEvent(self, event: QKeyEvent):
        # If the Esc key is pressed, close the application
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_Space:
            if not self.is_running:
                worker = Worker(self.iterateGridSnakes)
                worker.signal.success.connect(self._on_iterate_complete)
                QThreadPool.globalInstance().start(worker)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = CalibrationWindow(20, 30)
    sys.exit(app.exec_())

