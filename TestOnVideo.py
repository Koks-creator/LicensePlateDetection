from time import time
import cv2

from LicensePlateDetection.ocrTool.ocr_tool import OcrTool
from LicensePlateDetection.detector import Detector
from LicensePlateDetection.Pipelines.pipeline import PlateLicensePipeline
from LicensePlateDetection.PipelineTasks.tasks import PlateRoiTask, ApplyThreshTask, GetTextTask

t1 = PlateRoiTask(name="PlateRoi", description="Getting warpPerspective of license plate")
t2 = ApplyThreshTask(name="ApplyThreshold", description="Applying threshold in order to improve OCR results")
t3 = GetTextTask(name="GetText", description="Getting OCR results")


pipeline = PlateLicensePipeline(
    pipeline_name="Plate License Preprocessing",
    pipeline_description="Pipeline for preprocessing image and getting license plate text",
    tasks=[t1, t3],
    # tasks=[t1, t2, t3],
    log_level="ERROR"
)

detector = Detector(
    weights_file_path=r"./Model/yolov3_training_final.weights",
    config_file_path=r"./Model/yolov3_testing.cfg",
    classes_file_path=r"./Model/classes.txt",
    confidence_threshold=.1,
    nms_threshold=.1
)
ocr = OcrTool()

cap = cv2.VideoCapture(r"./Videos/1.mp4")

p_time = 0
while cap.isOpened():
    _, img = cap.read()

    detections = detector.detect(img)

    for index, detection in enumerate(detections):
        x1, y1 = detection.x, detection.y
        x2, y2 = detection.x + detection.w, detection.y + detection.h

        res_img, bbox, ocr_text = pipeline.execute_tasks(img, (x1, y1, detection.w, detection.h))
        if ocr_text == "":
            ocr_text = "NotFound"

        cv2.rectangle(img, (x1, y1), (x2, y2), (50, 0, 200), 2)
        cv2.putText(img, ocr_text, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 200, 0), 2)

        # cv2.imshow(f"res_img{index}", res_img)

    c_time = time()
    fps = int(1 / (c_time - p_time))
    p_time = c_time

    cv2.putText(img, f"FPS: {fps}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    cv2.imshow("Out3", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()