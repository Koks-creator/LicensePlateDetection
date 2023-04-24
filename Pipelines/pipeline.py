from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from typing import List, TypeVar


@dataclass
class PipelineTask(ABC):
    name: str
    description: str = ""

    @abstractmethod
    def run(self, *args, **kwargs):
        '''
        Put all processing here
        :return:
        '''
        pass


U = TypeVar('U', bound=PipelineTask)


@dataclass
class Pipeline(ABC):
    pipeline_name: str
    pipeline_description: str
    tasks: List[U]
    log_level: str = "ERROR"

    @abstractmethod
    def execute_tasks(self, *args, **kwargs):
        pass


class PlateLicensePipeline(Pipeline):
    def __init__(self, pipeline_name: str, pipeline_description: str, tasks, log_level: str = "ERROR"):
        U = TypeVar('U', bound=PipelineTask)

        self.pipeline_name = pipeline_name
        self.pipeline_description = pipeline_description
        self.tasks: List[U] = tasks

        self._format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(format=self._format)
        self.logger = logging.getLogger(self.pipeline_name)
        self.logger.setLevel(log_level.upper())

    def execute_tasks(self, image, bbox: tuple, stop_after_error: bool = True):
        ocr_res = ""

        for task in self.tasks:
            self.logger.info(f"Starting task: {task.name}")
            self.logger.debug(f"{task.name} parameters: {image.shape=}, {bbox=}, {ocr_res=}")

            try:
                image, bbox, ocr_res = task.run(image, bbox)
            except Exception as e:
                self.logger.error(f"Error in {task.name}: {e}")
                if stop_after_error:
                    self.logger.error(f"Stopping execution")
                    break

            self.logger.info(f"Task: {task.name} done")

        return image, bbox, ocr_res
