from dataclasses import dataclass
from typing import Union
import re
import easyocr
import numpy as np


@dataclass
class OcrTool:
    input_lang: str = "en"
    target_lang: str = "en"
    confidence_threshold: float = 0.1

    def __post_init__(self) -> None:
        self.reader = easyocr.Reader([self.input_lang, self.target_lang])

    def get_text_from_image(self, image: Union[str, np.array]) -> str:
        text_res = ""

        result = self.reader.readtext(image)
        for res in result:
            text, conf = res[1:]
            if conf > 0.1:
                text_res += text

        text_res = re.sub('[^A-Za-z0-9]+', '', text_res)

        return text_res
