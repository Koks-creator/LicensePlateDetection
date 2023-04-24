from typing import Tuple
from dataclasses import dataclass
import cv2
import numpy as np

from LicensePlateDetection.Pipelines.pipeline import PipelineTask
from LicensePlateDetection.ocrTool.ocr_tool import OcrTool


@dataclass
class PlateRoiTask(PipelineTask):

    def run(self, image: np.array, bbox: tuple, scale: int = 6) -> Tuple[np.array, tuple, str]:
        """
        :param image: bg image
        :param bbox:
        :param scale: resize scale
        :return:
        """
        x, y, w, h = bbox
        x1, y1 = x, y
        x2, y2 = x1 + w, y1 + h

        points_list = [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]
        width, height = points_list[3][0] - points_list[0][0], points_list[3][1] - points_list[0][1]

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
        plate = cv2.warpPerspective(image, matrix, (width, height))
        plate = cv2.resize(plate, None, fx=scale, fy=scale)

        return plate, bbox, ""


@dataclass
class ApplyThreshTask(PipelineTask):

    def run(self, image: np.array, bbox: tuple) -> Tuple[np.array, tuple, str]:
        res_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh_image = cv2.threshold(res_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        return thresh_image, bbox, ""


@dataclass
class GetTextTask(PipelineTask):
    ocr_threshold: float = 0.1

    def __post_init__(self):
        self.ocr = OcrTool()

    def run(self, image: np.array, bbox: tuple) -> Tuple[np.array, tuple, str]:
        text = self.ocr.get_text_from_image(image)

        return image, bbox, text
