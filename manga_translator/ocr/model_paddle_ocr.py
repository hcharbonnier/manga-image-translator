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
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en')  # Initialize PaddleOCR
        self.lang_classifier = PaddleClas(model_name="language_classification")  # Initialize PaddleClas

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

        perm = range(len(region_imgs))
        is_quadrilaterals = False
        if len(quadrilaterals) > 0 and isinstance(quadrilaterals[0][0], Quadrilateral):
            perm = sorted(range(len(region_imgs)), key=lambda x: region_imgs[x].shape[1])
            is_quadrilaterals = True

        texts = {}
        merged_idx = [[i] for i in range(len(region_imgs))]
        merged_quadrilaterals = quadrilaterals
        merged_region_imgs = []
        for q, d in merged_quadrilaterals:
            if d == 'h':
                merged_text_height = q.aabb.w
                merged_d = 'h'
            elif d == 'v':
                merged_text_height = q.aabb.h
                merged_d = 'h'
            merged_region_imgs.append(q.get_transformed_region(image, merged_d, merged_text_height))
        for idx in range(len(merged_region_imgs)):
            try:
                # Use PaddleOCR for OCR
                result = self.ocr.ocr(merged_region_imgs[idx], cls=True)
                print(f"OCR result: {result}")
                # Use PaddleClas for language detection
                lang_result = next(self.lang_classifier.predict(input_data=merged_region_imgs[idx]))
                detected_lang = lang_result['class_name']
                if self.logger:
                    self.logger.info(f"Detected language: {detected_lang}")
                    self.logger.info(f"OCR result: {result}")  # Log the OCR result
                if not result or not isinstance(result, list):
                    raise ValueError("Invalid OCR result format")
                # Extract and concatenate the recognized text
                texts[idx] = " ".join([line[1][0] for line in result[0] if line and isinstance(line, list) and len(line) > 1])
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error during OCR: {e}")
                texts[idx] = ""

        ix = 0
        out_regions = {}
        print(f"Texts: {texts}")
        for indices in chunks(perm, max_chunk_size):
            N = len(indices)
            widths = [region_imgs[i].shape[1] for i in indices]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((N, text_height, max_width, 3), dtype=np.uint8)
            idx_keys = []
            for i, idx in enumerate(indices):
                idx_keys.append(idx)
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
                cur_region = quadrilaterals[indices[i]][0]
                cur_region.text = texts[indices[i]]
                cur_region.prob = 1.0  # Set a default probability
                out_regions[idx_keys[i]] = cur_region

        output_regions = []
        for i, nodes in enumerate(merged_idx):
            total_logprobs = 0
            fg_r = []
            fg_g = []
            fg_b = []
            bg_r = []
            bg_g = []
            bg_b = []

            for idx in nodes:
                if idx not in out_regions:
                    continue

                total_logprobs += np.log(out_regions[idx].prob)
                fg_r.append(out_regions[idx].fg_r)
                fg_g.append(out_regions[idx].fg_g)
                fg_b.append(out_regions[idx].fg_b)
                bg_r.append(out_regions[idx].bg_r)
                bg_g.append(out_regions[idx].bg_g)
                bg_b.append(out_regions[idx].bg_b)

            prob = np.exp(total_logprobs / len(nodes))
            fr = round(np.mean(fg_r))
            fg = round(np.mean(fg_g))
            fb = round(np.mean(fg_b))
            br = round(np.mean(bg_r))
            bg = round(np.mean(bg_g))
            bb = round(np.mean(bg_b))

            txt = texts[nodes[0]]  # Ensure correct indexing
            if self.logger:
                self.logger.info(f'prob: {prob} {txt} fg: ({fr}, {fg}, {fb}) bg: ({br}, {bg}, {bb})')
            cur_region = merged_quadrilaterals[i][0]
            if isinstance(cur_region, Quadrilateral):
                cur_region.text = txt
                cur_region.prob = prob
                cur_region.fg_r = fr
                cur_region.fg_g = fg
                cur_region.fg_b = fb
                cur_region.bg_r = br
                cur_region.bg_g = bg
                cur_region.bg_b = bb
            else:  # TextBlock
                cur_region.text.append(txt)
                cur_region.update_font_colors(np.array([fr, fg, fb]), np.array([br, bg, bb]))
            output_regions.append(cur_region)

        if is_quadrilaterals:
            return output_regions
        return textlines
