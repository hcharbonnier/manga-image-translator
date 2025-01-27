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
        text_height = getattr(config, 'text_height', 48)  # Use default text height if not specified
        
        # Generate Text Direction
        for textline in textlines:
            direction = "horizontal"  # Placeholder for actual direction determination logic
            # Determine the direction of the text region
            if textline.width() > textline.height():
                direction = "horizontal"
            else:
                direction = "vertical"
            # ...existing code...

        # Transform Regions
        transformed_regions = []
        for textline in textlines:
            transformed_region = textline  # Placeholder for actual transformation logic
            # Transform each text region to a standard size
            transformed_region = textline.get_transformed_region(image, direction, text_height)
            transformed_regions.append(transformed_region)
            # ...existing code...

        # Recognize Text
        for region in transformed_regions:
            # Convert region to a valid image type for PaddleClas
            region_image = Image.fromarray(region) if isinstance(region, np.ndarray) else region
            
            # Use PaddleClas to detect language
            lang_result = list(self.lang_classifier.predict(region_image))
            detected_lang = lang_result[0]['label']
            
            # Use PaddleOCR to recognize text
            ocr_result = self.ocr.ocr(region, cls=False)
            for line in ocr_result:
                text, prob = line[1][0], line[1][1]
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
