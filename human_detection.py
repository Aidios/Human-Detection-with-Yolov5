import cv2
import torch
import numpy as np
import dlib
import os
from sklearn.metrics import confusion_matrix

class YoloDetector:
    def __init__(self, weights='C:/VS Code/Python/yolov5/models/weights/yolov5s.pt', device='cpu'):
        self.weights = weights
        self.device = device
        self.model = self.load_model()
        self.students = {}
        self.student_id_counter = 1
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor('C:/VS Code/Python/yolov5/shape_predictor_68_face_landmarks.dat')
        self.face_recognition_model = dlib.face_recognition_model_v1('C:/VS Code/Python/yolov5/dlib_face_recognition_resnet_model_v1(1).dat')

        self.door_rect = (200, 0, 450, 475)
        self.entered_students = set()
        self.exited_students = set()

        self.correct_detections = 0  # Doğru yüz tanıma sayısı
        self.total_detections = 0    # Toplam yüz tespit sayısı

        self.log_file_path = os.path.join(os.path.dirname(__file__), 'student_log.txt')
        print(f"Log file path: {self.log_file_path}")

    def load_model(self):
        repo_or_dir = 'C:/VS Code/Python/yolov5'
        model = torch.hub.load(repo_or_dir, 'custom', path=self.weights, source='local')
        return model.to(self.device).eval()

    def detect_from_camera(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error opening video stream")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Kamerayı dikey eksende çevir
            frame = cv2.flip(frame, 0)

            results = self.model(frame)
            self.process_detections(results, frame)

            # Kapı alanını çerçevele
            x1, y1, x2, y2 = self.door_rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            self.check_entries_and_exits()

            # Kümeleme doğruluğu ve yüz tanıma doğruluğunu ekrana yazdır
            accuracy_text = f"Recognition Accuracy: {self.get_recognition_accuracy():.2f}%"
            cv2.putText(frame, accuracy_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Camera Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_detections(self, results, frame):
        detected_students = []

        for detection in results.xyxy[0]:
            label = int(detection[-1])
            if label == 0:  # Sadece kişi etiketini kontrol et
                x1, y1, x2, y2 = map(int, detection[:4])
                face_region = frame[y1:y2, x1:x2]

                dlib_faces = self.face_detector(face_region)
                if len(dlib_faces) > 0:
                    shape = self.shape_predictor(face_region, dlib_faces[0])
                    face_encoding = np.array(self.face_recognition_model.compute_face_descriptor(frame, shape))

                    student_id = self.get_face_id(face_encoding)
                    detected_students.append(student_id)

                    if student_id in self.students:
                        self.correct_detections += 1  # Doğru tanıma
                        self.students[student_id]['bbox'] = (x1, y1, x2, y2)
                    else:
                        self.students[student_id] = {'bbox': (x1, y1, x2, y2), 'misses': 0}

                    self.total_detections += 1  # Toplam tespit
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'Student {student_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        self.remove_lost_students(detected_students)

    def get_face_id(self, face_encoding, threshold=0.6):
        for student_id, data in self.students.items():
            if self.compare_face_encoding(data['encoding'], face_encoding, threshold):
                return student_id

        new_student_id = str(self.student_id_counter)
        self.students[new_student_id] = {'encoding': face_encoding.tolist(), 'bbox': [], 'misses': 0}
        self.student_id_counter += 1
        return new_student_id

    def compare_face_encoding(self, known_encoding, face_encoding, threshold):
        return np.linalg.norm(np.array(known_encoding) - np.array(face_encoding)) < threshold

    def remove_lost_students(self, detected_students):
        lost_students = [sid for sid in self.students if sid not in detected_students]
        for student_id in lost_students:
            self.students[student_id]['misses'] += 1
            if self.students[student_id]['misses'] > 5:
                del self.students[student_id]

    def check_entries_and_exits(self):
        x1, y1, x2, y2 = self.door_rect
        for student_id, data in self.students.items():
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = data['bbox']
            if (bbox_x1 >= x1 and bbox_x2 <= x2 and bbox_y1 >= y1 and bbox_y2 <= y2):
                if student_id not in self.entered_students:
                    self.entered_students.add(student_id)
                    with open(self.log_file_path, 'a') as log_file:
                        log_file.write(f'Student {student_id} giriş yaptı\n')
                    print(f'Student {student_id} giriş yaptı')
            else:
                if student_id in self.entered_students:
                    self.exited_students.add(student_id)
                    with open(self.log_file_path, 'a') as log_file:
                        log_file.write(f'Student {student_id} çıkış yaptı\n')
                    print(f'Student {student_id} çıkış yaptı')
                    self.entered_students.remove(student_id)

    def get_recognition_accuracy(self):
        if self.total_detections == 0:
            return 0.0
        return (self.correct_detections / self.total_detections) * 100

if __name__ == "__main__":
    detector = YoloDetector()
    detector.detect_from_camera()
