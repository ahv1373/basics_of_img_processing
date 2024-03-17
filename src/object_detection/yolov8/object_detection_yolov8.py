# This code is for inferecne and prediction using a yolov8 model.
import glob
import time
import cv2
import numpy as np

from ultralytics import YOLO


class YOLOV8ObjectDetection:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.colors = None
        self.classes = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None  # Do not forget to load the model using the load method

    def load(self):
        self.model = YOLO(self.model_path)

        print("[INFO] Model loaded successfully.")

    def __detect__(self, frame: np.array):
        results = self.model(frame)
        return results

    def visualize(self, frame, bboxes_list, class_index_list, scores_list):
        for bbox, class_index, confidence in zip(bboxes_list, class_index_list, scores_list):
            start_x, start_y, end_x, end_y = bbox
            label = "{}: {:.2f}%".format(self.classes[class_index], confidence * 100)
            # draw the prediction on the frame
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), self.colors[class_index], 2)
            y = start_y - 15 if start_y - 15 > 15 else start_y + 15
            cv2.putText(frame, label, (start_x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, self.colors[class_index], 2)

    def exec(self, frame: np.array):
        results = self.__detect__(frame)
        if self.classes is None:
            self.classes = results[0].names
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        bounding_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        confidence_scores = results[0].boxes.conf.cpu().numpy()
        return bounding_boxes, class_indices, confidence_scores


if __name__ == "__main__":
    DATA_DIR = "/home/ahv/PycharmProjects/basics_of_img_processing/src/dataset/london_street"
    images_list = glob.glob(DATA_DIR + "/*.jpg")
    images_list.sort()
    object_detector = YOLOV8ObjectDetection(model_path="yolov8_custom.pt")
    object_detector.load()

    # for img_name in images_list:
    #     frame = np.array(cv2.imread(img_name))
    #     bounding_boxes, class_indices, confidence_scores = object_detector.exec(frame)
    #     object_detector.visualize(frame, bounding_boxes, class_indices, confidence_scores)
    #     cv2.imshow("Frame", frame)
    #     key = cv2.waitKey(0)
    #     if key == ord("q"):
    #         break
    #     elif key == ord("s"):
    #         cv2.imwrite(f"{img_name}", frame)

    # process using webcam
    webcam = cv2.VideoCapture(0)
    average_fps = 0
    while True:
        ret, frame = webcam.read()
        start_time = time.time()
        bounding_boxes, class_indices, confidence_scores = object_detector.exec(frame)
        inference_time = time.time() - start_time
        object_detector.visualize(frame, bounding_boxes, class_indices, confidence_scores)
        average_fps = (0.9 * average_fps) + (0.1 * (1.0 / inference_time))
        cv2.putText(frame, f"Average FPS: {average_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_COMPLEX,
                    0.9, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

