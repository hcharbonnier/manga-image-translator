import itertools
import math
from typing import Callable, List, Set, Optional, Tuple, Union
from collections import defaultdict, Counter
import os
import cv2
from PIL import Image
import numpy as np
import einops
import networkx as nx
from shapely.geometry import Polygon

import torch

import shutil

from .common import OfflineOCR
from ..config import OcrConfig
from ..textline_merge import split_text_region
from ..utils import TextBlock, Quadrilateral, quadrilateral_can_merge_region, chunks
from ..utils.generic import AvgMeter

from paddleocr import PaddleOCR
from paddleclas import PaddleClas

class ModelPaddleOCR(OfflineOCR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = kwargs.get('logger', None)  # Ensure logger is defined

    async def _load(self, device: str):
        self.device = device
        self.use_gpu = device in ['cuda', 'mps']
        
        # Load the OCR model and its dictionary
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en')
        
        # Load the language classifier
        self.lang_classifier = PaddleClas(model_name="language_classification")
        print("Language classifier loaded and set to evaluation mode.")

    async def _unload(self):
        # Clean up by deleting the model and related objects from memory
        del self.ocr
        del self.lang_classifier
        torch.cuda.empty_cache()
        print("Models unloaded and memory cleaned up.")

    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False, ignore_bubble: int = 0) -> List[TextBlock]:
        results = []
        
        # Generate Text Direction
        for textline in textlines:
            # Determine the direction of the text region
            direction = "horizontal"  # Placeholder for actual direction determination logic
            # ...existing code...

        # Transform Regions
        transformed_regions = []
        for textline in textlines:
            # Transform each text region to a standard size
            transformed_region = textline  # Placeholder for actual transformation logic
            transformed_regions.append(transformed_region)
            # ...existing code...

        # Recognize Text
        for region in transformed_regions:
            # Use PaddleClas to detect language
            lang_result = self.lang_classifier.predict(region)
            detected_lang = lang_result[0]['label']
            
            # Use PaddleOCR to recognize text
            ocr_result = self.ocr.ocr(region, cls=False)
            for line in ocr_result:
                text, prob = line[1][0], line[1][1]
                # Log the recognized text and its attributes
                if verbose:
                    print(f"Recognized text: {text} with probability: {prob}")
                
                # Post-process Recognized Text
                font_color = (255, 255, 255)  # Placeholder for actual font color calculation
                text_block = TextBlock(
                    text=text,
                    probability=prob,
                    font_color=font_color,
                    position=region
                )
                results.append(text_block)

        # Return Results
        print(f"Returning {len(results)} text blocks.")
        print("Results:", results)
        return results
