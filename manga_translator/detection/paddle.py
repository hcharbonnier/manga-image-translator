import os
import shutil
import numpy as np
import cv2
from paddleocr import PaddleOCR
from typing import List, Tuple
from paddleocr_convert import PaddleOCRModelConvert
from rapidocr_onnxruntime import RapidOCR

from .common import OfflineDetector
from ..utils import TextBlock, Quadrilateral
from ..utils.inference import ModelWrapper

MODEL = None

class PaddleDetector(OfflineDetector, ModelWrapper):
    _MODEL_MAPPING = {
        'det': {
            'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar',
            'hash': '0c0e4fc2ef31dcfbb45fb8d29bd8e702ec55a240d62c32ff814270d8be6e6179',
            # 'archive': {
            #     'ch_PP-OCRv4_det_server_infer/inference.pdiparams': 'ch_PP-OCRv4_det_server_infer/',
            #     'ch_PP-OCRv4_det_server_infer/inference.pdiparams.info': 'ch_PP-OCRv4_det_server_infer/',
            #     'ch_PP-OCRv4_det_server_infer/inference.pdmodel': 'ch_PP-OCRv4_det_server_infer/',
            # },
        },
        'rec': {
            'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar',
            'hash': '830ea228e20c2b30c4db9666066c48512f67a63f5b1a32d0d33dc9170040ce7d',
            # 'archive': {
            #     'ch_PP-OCRv4_rec_infer/inference.pdiparams': 'ch_PP-OCRv4_rec_infer/',
            #     'ch_PP-OCRv4_rec_infer/inference.pdiparams.info': 'ch_PP-OCRv4_rec_infer/',
            #     'ch_PP-OCRv4_rec_infer/inference.pdmodel': 'ch_PP-OCRv4_rec_infer/',
            #},
        },
        'cls': {
            'url': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
            'hash': '507352585040d035da3b1e6374694ad679a850acb0a36a8d0d47984176357717',
            # 'archive': {
            #     'ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams': 'ch_ppocr_mobile_v2.0_cls_infer/',
            #     'ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel': 'ch_ppocr_mobile_v2.0_cls_infer/',
            # },
        }
    }

    def __init__(self, *args, **kwargs):
        ModelWrapper.__init__(self)
        super().__init__(*args, **kwargs)

    async def convert(self):
        converter = PaddleOCRModelConvert()
        for model in self._MODEL_MAPPING.values():
            url = model['url']
            archive = url.split('/')[-1]
            converted_folder = archive.split('.')[0]
            model_path = self.model_dir + '/' + archive
            save_dir =  self.model_dir + '/onnx/'
            if not os.path.exists(save_dir + '/' + converted_folder):
                txt_path = 'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/release/2.9/ppocr/utils/ppocr_keys_v1.txt'
                converter(model_path, save_dir, txt_path)

    async def _load(self, device: str, text_threshold: float, box_threshold: float, unclip_ratio: float, invert: bool = False, verbose: bool = False):
        print("Loading PaddleOCR model")
        await self.download()
        print("Converting PaddleOCR model to ONNX")
        await self.convert()
        self.device = device
        self.text_threshold = text_threshold
        self.box_threshold = box_threshold
        self.unclip_ratio = unclip_ratio
        self.invert = invert
        self.verbose = verbose
        if device in ['cuda', 'mps']:
            self.use_gpu = True
        else:
            self.use_gpu = False
        global MODEL
        MODEL = RapidOCR(
            rec_model_path=self.model_dir+'/onnx/ch_PP-OCRv4_rec_infer/ch_PP-OCRv4_rec_infer.onnx',
            det_model_path=self.model_dir+'/onnx/ch_PP-OCRv4_det_server_infer/ch_PP-OCRv4_det_server_infer.onnx',
            cls_model_path=self.model_dir+'/onnx/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx',
            det_use_cuda=self.use_gpu,
            cls_use_cuda=self.use_gpu,
            rec_use_cuda=self.use_gpu,
            det_db_thresh=self.text_threshold,
            det_db_box_thresh=self.box_threshold,
            det_db_unclip_ratio=self.unclip_ratio,
            invert=self.invert,
            verbose=self.verbose,
            det=True,
            rec=False,
            cls=False
        )

    async def _unload(self):
        global MODEL
        MODEL = None

    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, verbose: bool = False):
        global MODEL
        #result = MODEL.ocr(image, det=True, rec=False)
        result = MODEL.engine(image)

        textlines = []

        # Parse OCR results and filter by text threshold
        for line in result[0]:
            points = np.array(line).astype(np.int32)
            # paddleocr does not return score, so we use a fixed value: 1
            textlines.append(Quadrilateral(points, '', 1))

        # Create a binary mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for textline in textlines:
            cv2.fillPoly(mask, [textline.pts], color=255)

        # Additional polygon refinement
        refined_polys = []
        for textline in textlines:
            poly = cv2.minAreaRect(textline.pts)
            box = cv2.boxPoints(poly)
            box = np.int0(box)
            refined_polys.append(np.roll(box, 2, axis=0))  # Ensure clockwise order

        # Update mask with refined polygons
        for poly in refined_polys:
            mask = cv2.fillPoly(mask, [poly], color=255)

        # Return textlines with refined polygons
        textlines = [Quadrilateral(poly, '', 1) for poly, textline in zip(refined_polys, textlines)]

        return textlines, mask, None
