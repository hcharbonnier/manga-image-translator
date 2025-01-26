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

class ModelPaddleOCR(OfflineOCR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = kwargs.get('logger', None)  # Ensure logger is defined
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en')  # Initialize PaddleOCR
        self.lang_map = {
            'en': 'en',
            'es': 'french',  # PaddleOCR does not support Spanish directly
            'fr': 'french',
            'de': 'german',
            'it': 'italian',  # PaddleOCR does not support Italian directly
            'pt': 'portuguese',  # PaddleOCR does not support Portuguese directly
            'ru': 'russian',  # PaddleOCR does not support Russian directly
            'ja': 'japan',
            'ko': 'korean',
            'zh-cn': 'ch',  # Simplified Chinese
            'zh-tw': 'ch',  # Traditional Chinese
            'ar': 'arabic',  # PaddleOCR does not support Arabic directly
            'hi': 'hindi',  # PaddleOCR does not support Hindi directly
            'bn': 'bengali',  # PaddleOCR does not support Bengali directly
            'pa': 'punjabi',  # PaddleOCR does not support Punjabi directly
            'jv': 'javanese',  # PaddleOCR does not support Javanese directly
            'ms': 'malay',  # PaddleOCR does not support Malay directly
            'id': 'indonesian',  # PaddleOCR does not support Indonesian directly
            'vi': 'vietnamese',  # PaddleOCR does not support Vietnamese directly
            'th': 'thai',  # PaddleOCR does not support Thai directly
            # Add other mappings as needed
        }

    async def _load(self, device: str):
        self.device = device
        self.use_gpu = device in ['cuda', 'mps']

    async def _unload(self):
        pass
    
    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False, ignore_bubble: int = 0) -> List[TextBlock]:
        text_height = 48
        max_chunk_size = 16

        quadrilaterals = list(self._generate_text_direction(textlines))
        region_imgs = [q.get_transformed_region(image, d, text_height) for q, d in quadrilaterals]
        out_regions = []

        perm = range(len(region_imgs))
        is_quadrilaterals = False
        if len(quadrilaterals) > 0 and isinstance(quadrilaterals[0][0], Quadrilateral):
            perm = sorted(range(len(region_imgs)), key=lambda x: region_imgs[x].shape[1])
            is_quadrilaterals = True

        ix = 0
        for indices in chunks(perm, max_chunk_size):
            N = len(indices)
            widths = [region_imgs[i].shape[1] for i in indices]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((N, text_height, max_width, 3), dtype=np.uint8)
            for i, idx in enumerate(indices):
                W = region_imgs[idx].shape[1]
                tmp = region_imgs[idx]
                region[i, :, :W, :] = tmp
                if verbose:
                    os.makedirs('result/ocrs/', exist_ok=True)
                    if quadrilaterals[idx][1] == 'v':
                        cv2.imwrite(f'result/ocrs/{ix}.png', cv2.rotate(cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE))
                    else:
                        cv2.imwrite(f'result/ocrs/{ix}.png', cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR))
                ix += 1

            for i in range(N):
                try:
                    # Use PaddleOCR for OCR
                    result = self.ocr.ocr(region[i], cls=True)
                    detected_lang = 'en'  # PaddleOCR does not support language detection
                    if self.logger:
                        self.logger.info(f"Detected language: {detected_lang}")
                        self.logger.info(f"OCR result: {result}")  # Log the OCR result
                    txt = " ".join([line[1][0] for line in result])
                    prob = 1.0  # Set a default probability
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error during OCR: {e}")
                    txt = ""
                    prob = 0.0

                cur_region = quadrilaterals[indices[i]][0]
                if isinstance(cur_region, Quadrilateral):
                    cur_region.text = txt
                    cur_region.prob = prob
                    cur_region.fg_r = 0
                    cur_region.fg_g = 0
                    cur_region.fg_b = 0
                    cur_region.bg_r = 255
                    cur_region.bg_g = 255
                    cur_region.bg_b = 255
                else:
                    cur_region.text.append(txt)
                    cur_region.update_font_colors(np.array([0, 0, 0]), np.array([255, 255, 255]))

                out_regions.append(cur_region)

        if is_quadrilaterals:
            return out_regions
        return textlines
