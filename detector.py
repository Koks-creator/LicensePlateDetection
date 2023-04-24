from dataclasses import dataclass
import cv2
import numpy as np

from LicensePlateDetection.ocrTool.ocr_tool import OcrTool


@dataclass
class DetectionData:
    x: int
    y: int
    w: int
    h: int
    class_name: str
    detections_conf: float
    color: list


@dataclass
class Detector:
    weights_file_path: str
    config_file_path: str
    classes_file_path: str
    image_width: int = 416
    image_height: int = 416
    confidence_threshold: float = 0.3
    nms_threshold: float = 0.3

    def __post_init__(self) -> None:
        self.net = cv2.dnn.readNet(self.weights_file_path, self.config_file_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        with open(self.classes_file_path) as f:
            self.classes = f.read().splitlines()

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect(self, img: np.array) -> list:
        """
        :param img: input img
        :return: list of tuples containing the following data: x, y, w, h, class_name, confidence, class_color
        """
        bbox = []
        class_ids = []
        confs = []

        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (self.image_width, self.image_height), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                # print(class_id)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int((detection[0] * width) - w/2)
                    y = int((detection[1] * height) - h/2)

                    bbox.append([x, y, w, h])
                    class_ids.append(class_id)
                    confs.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(bbox, confs, self.confidence_threshold, self.nms_threshold)

        detections_list = []
        for i in indexes:
            i = i[0]

            box = bbox[i]
            x, y, w, h = box
            class_name = self.classes[class_ids[i]].capitalize()
            conf = confs[i]
            class_color = [int(c) for c in self.colors[class_ids[i]]]

            detections_list.append(DetectionData(x, y, w, h, class_name, conf, class_color))

        return detections_list


if __name__ == '__main__':
    detector = Detector(
        weights_file_path=r"Model/yolov3_training_final.weights",
        config_file_path=r"Model/yolov3_testing.cfg",
        classes_file_path=r"Model/classes.txt",
        confidence_threshold=.1,
        nms_threshold=.1
    )
    ocr_tool = OcrTool()

    img = cv2.imread(r"testImages/tr.png")

    detections = detector.detect(img)
    for detection in detections:
        x1, y1 = detection.x, detection.y
        x2, y2 = detection.x + detection.w, detection.y + detection.h

        cv2.rectangle(img, (x1, y1), (x2, y2), detection.color, 2)

        cv2.putText(img, f"{detection.class_name} {int(round(detection.detections_conf, 2) * 100)}%",
                    (x1, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 1.5, detection.color, 2)

        width, height = detection.w, detection.h

        roi_x1, roi_y1 = x1, y1
        roi_x2, roi_y2 = x2, y1
        roi_x3, roi_y3 = x1, y2
        roi_x4, roi_y4 = x2, y2

        cv2.circle(img, (roi_x1, roi_y1), 5, (255, 0, 0), -1)
        cv2.circle(img, (roi_x2, roi_y2), 5, (255, 0, 0), -1)
        cv2.circle(img, (roi_x3, roi_y3), 5, (255, 0, 0), -1)
        cv2.circle(img, (roi_x4, roi_y4), 5, (255, 0, 0), -1)

        points_list = [(roi_x1, roi_y1), (roi_x2, roi_y2), (roi_x3, roi_y3), (roi_x4, roi_y4)]

        pts1 = np.float32([
            [points_list[0][0], points_list[0][1]],
            [points_list[1][0], points_list[1][1]],
            [points_list[2][0], points_list[2][1]],
            [points_list[3][0], points_list[3][1]],
        ])

        pts2 = np.float32([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height],
        ])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        out = cv2.warpPerspective(img, matrix, (width, height))

        text_res = ocr_tool.get_text_from_image(out)  # moze threshold adaptacyjny?
        print(text_res)  # czyscic z bialych znakow

        #
        # cv2.imshow("Out", out)
        # break

    cv2.imshow("resImage", img)
    cv2.waitKey(0)

