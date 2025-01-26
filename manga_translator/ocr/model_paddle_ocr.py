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
from ..utils import TextBlock, Quadrilateral, quadrilateral_can_merge_region, chunks, is_ignore
from ..utils.generic import AvgMeter
from ..utils.inference import ModelWrapper

from paddleocr import PaddleOCR
from paddleclas import PaddleClas

MODEL = None

class ModelPaddleOCR(OfflineOCR,ModelWrapper):
    _MODEL_MAPPING = {
        'det': {
            'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar',
            'hash': '0c0e4fc2ef31dcfbb45fb8d29bd8e702ec55a240d62c32ff814270d8be6e6179',
            'archive': {
                'ch_PP-OCRv4_det_server_infer/inference.pdiparams': 'ch_PP-OCRv4_det_server_infer/',
                'ch_PP-OCRv4_det_server_infer/inference.pdiparams.info': 'ch_PP-OCRv4_det_server_infer/',
                'ch_PP-OCRv4_det_server_infer/inference.pdmodel': 'ch_PP-OCRv4_det_server_infer/',
            },
        },
        'rec': {
            'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar',
            'hash': '830ea228e20c2b30c4db9666066c48512f67a63f5b1a32d0d33dc9170040ce7d',
            'archive': {
                'ch_PP-OCRv4_rec_infer/inference.pdiparams': 'ch_PP-OCRv4_rec_infer/',
                'ch_PP-OCRv4_rec_infer/inference.pdiparams.info': 'ch_PP-OCRv4_rec_infer/',
                'ch_PP-OCRv4_rec_infer/inference.pdmodel': 'ch_PP-OCRv4_rec_infer/',
            },
        },
        'cls': {
            'url': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
            'hash': '507352585040d035da3b1e6374694ad679a850acb0a36a8d0d47984176357717',
            'archive': {
                'ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams': 'ch_ppocr_mobile_v2.0_cls_infer/',
                'ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel': 'ch_ppocr_mobile_v2.0_cls_infer/',
            },
        },
        'clas' :{
            'url': 'https://paddleclas.bj.bcebos.com/models/PULC/inference/language_classification_infer.tar',
            'hash': '0273e53013731b6d2418a801a7acdad8e609b64b1e06b17cb75cdc95343115b6',
            'archive': {
                'language_classification_infer/inference.pdiparams': 'language_classification_infer/',
                'language_classification_infer/inference.pdiparams.info': 'language_classification_infer/',
                'language_classification_infer/inference.pdmodel': 'language_classification_infer/',
            },
        }
    }

    def __init__(self, *args, **kwargs):
        ModelWrapper.__init__(self)
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        await self.download()
        self.device = device
        if device in ['cuda', 'mps']:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.lang_classifier = PaddleClas(inference_model_dir=self.model_dir+"/language_classification_infer/")  # Initialize PaddleClas
        global MODEL
        MODEL = PaddleOCR(
            use_gpu=self.use_gpu,
            use_angle_cls=False,
            det_model_dir=self.model_dir+'/ch_PP-OCRv4_det_server_infer',
            rec_model_dir=self.model_dir+'/ch_PP-OCRv4_rec_infer',
            cls_model_dir=self.model_dir+'/ch_ppocr_mobile_v2.0_cls_infer',
            det=False,
            rec=True,
            cls=False,
        )

    async def _unload(self):
        global MODEL
        MODEL = None

    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False, ignore_bubble: int = 0) -> List[TextBlock]:
        global MODEL

        text_height = 32
        max_chunk_size = 16

        quadrilaterals = list(self._generate_text_direction(textlines))
        region_imgs = [q.get_transformed_region(image, d, text_height) for q, d in quadrilaterals]
        out_regions = []

        perm = range(len(region_imgs))
        is_quadrilaterals = False
        if len(quadrilaterals) > 0 and isinstance(quadrilaterals[0][0], Quadrilateral):
            perm = sorted(range(len(region_imgs)), key = lambda x: region_imgs[x].shape[1])
            is_quadrilaterals = True

        ix = 0
        for indices in chunks(perm, max_chunk_size):
            # Filter valid images and record their original indices
            valid_indices = []
            valid_region_imgs = []
            for idx in indices:
                if ignore_bubble >= 1 and ignore_bubble <= 50 and is_ignore(region_imgs[idx], ignore_bubble):
                    ix += 1
                    continue
                valid_indices.append(idx)
                valid_region_imgs.append(region_imgs[idx])
                if verbose:
                    os.makedirs('result/ocrs/', exist_ok=True)
                    if quadrilaterals[idx][1] == 'v':
                        cv2.imwrite(f'result/ocrs/{ix}.png', cv2.rotate(cv2.cvtColor(region_imgs[idx], cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE))
                    else:
                        cv2.imwrite(f'result/ocrs/{ix}.png', cv2.cvtColor(region_imgs[idx], cv2.COLOR_RGB2BGR))
                ix += 1
            if not valid_region_imgs:
                continue
            # Assemble valid_region_imgs into a uniform batch
            widths = [img.shape[1] for img in valid_region_imgs]
            max_width = 4 * (max(widths) + 7) // 4
            N = len(valid_region_imgs)
            region_batch = np.zeros((N, text_height, max_width, 3), dtype=np.uint8)
            for j, img in enumerate(valid_region_imgs):
                W = img.shape[1]
                region_batch[j, :, :W, :] = img

            # Process OCR for the valid batch
            for j, idx in enumerate(valid_indices):
                result = MODEL.ocr(region_batch[j])
                if not result or not result[0]:
                    continue
                for line in result[0]:
                    if not line or not line[1] or not line[1][0] or not line[1][1]:
                        continue
                    txt = line[1][0]
                    prob = line[1][1]
                    if prob < 0.7:
                        continue
                    cur_region = quadrilaterals[idx][0]
                    if isinstance(cur_region, Quadrilateral):
                        cur_region.text = txt
                        cur_region.prob = prob
                        out_regions.append(cur_region)
                    else:
                        cur_region.text.append(txt)
                        out_regions.append(cur_region)

        if is_quadrilaterals:
            return out_regions
        return textlines