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
        },
    }

    def __init__(self, *args, **kwargs):
        ModelWrapper.__init__(self)
        super().__init__(*args, **kwargs)

    async def convert(self):
        converter = PaddleOCRModelConvert()
        model_path = self.model_dir + '/ch_PP-OCRv4_det_server_infer.tar'
        save_dir =  self.model_dir + '/onnx/'
        if not os.path.exists(self.model_dir + '/onnx/ch_PP-OCRv4_det_server_infer'):
            converter(model_path, save_dir)

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
        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        MODEL = RapidOCR(
            det_model_path=self.model_dir+'/onnx/ch_PP-OCRv4_det_server_infer/ch_PP-OCRv4_det_server_infer.onnx',
            det_use_cuda=self.use_gpu,
            det_db_thresh=self.text_threshold,
            det_db_box_thresh=self.box_threshold,
            det_db_unclip_ratio=self.unclip_ratio,
            verbose=self.verbose,
            no_rec=True,
            no_cls=True,
        )

    async def _unload(self):
        global MODEL
        MODEL = None

    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, verbose: bool = False):
        global MODEL
        print("Running PaddleOCR model")
        result = MODEL(image)
        print("PaddleOCR model finished")

        textlines = []
        with open('result.log', 'w') as f:
            f.write(str(result))

        # Parse OCR results and filter by text threshold
        for line in result[0]:
            points = np.array(line[0]).astype(np.int32)
            score = line[2]
            # paddleocr does not return score, so we use a fixed value: 1
            textlines.append(Quadrilateral(points, '', score))
            # wrire at the end of a a file
            with open('textlines.txt', 'a') as f:
                f.write(str(points) + '\n')

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
