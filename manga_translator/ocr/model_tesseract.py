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

import pytesseract

from .common import OfflineOCR
from ..config import OcrConfig
from ..textline_merge import split_text_region
from ..utils import TextBlock, Quadrilateral, quadrilateral_can_merge_region, chunks
from ..utils.generic import AvgMeter

async def merge_bboxes(bboxes: List[Quadrilateral], width: int, height: int) -> Tuple[List[Quadrilateral], int]:
    # step 1: divide into multiple text region candidates
    G = nx.Graph()
    for i, box in enumerate(bboxes):
        G.add_node(i, box=box)
    for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2):
        # if quadrilateral_can_merge_region_coarse(ubox, vbox):
        if quadrilateral_can_merge_region(ubox, vbox, aspect_ratio_tol=1.3, font_size_ratio_tol=2,
                                          char_gap_tolerance=1, char_gap_tolerance2=3):
            G.add_edge(u, v)

    # step 2: postprocess - further split each region
    region_indices: List[Set[int]] = []
    for node_set in nx.algorithms.components.connected_components(G):
         region_indices.extend(split_text_region(bboxes, node_set, width, height))

    # step 3: return regions
    merge_box = []
    merge_idx = []
    for node_set in region_indices:
    # for node_set in nx.algorithms.components.connected_components(G):
        nodes = list(node_set)
        txtlns: List[Quadrilateral] = np.array(bboxes)[nodes]

        # majority vote for direction
        dirs = [box.direction for box in txtlns]
        majority_dir_top_2 = Counter(dirs).most_common(2)
        if len(majority_dir_top_2) == 1 :
            majority_dir = majority_dir_top_2[0][0]
        elif majority_dir_top_2[0][1] == majority_dir_top_2[1][1] : # if top 2 have the same counts
            max_aspect_ratio = -100
            for box in txtlns :
                if box.aspect_ratio > max_aspect_ratio :
                    max_aspect_ratio = box.aspect_ratio
                    majority_dir = box.direction
                if 1.0 / box.aspect_ratio > max_aspect_ratio :
                    max_aspect_ratio = 1.0 / box.aspect_ratio
                    majority_dir = box.direction
        else :
            majority_dir = majority_dir_top_2[0][0]

        # sort textlines
        if majority_dir == 'h':
            nodes = sorted(nodes, key=lambda x: bboxes[x].centroid[1])
        elif majority_dir == 'v':
            nodes = sorted(nodes, key=lambda x: -bboxes[x].centroid[0])
        txtlns = np.array(bboxes)[nodes]
        # yield overall bbox and sorted indices
        merge_box.append(txtlns)
        merge_idx.append(nodes)
    
    return_box = []
    for bbox in merge_box:
        if len(bbox) == 1:
            return_box.append(bbox[0])
        else:
            prob = [q.prob for q in bbox]
            prob = sum(prob)/len(prob)
            base_box = bbox[0]
            for box in bbox[1:]:
                min_rect = np.array(Polygon([*base_box.pts, *box.pts]).minimum_rotated_rectangle.exterior.coords[:4])
                base_box = Quadrilateral(min_rect, '', prob)
            return_box.append(base_box)
    return return_box, merge_idx

class ModelTesseractOCR(OfflineOCR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = kwargs.get('logger', None)  # Ensure logger is defined

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
            perm = sorted(range(len(region_imgs)), key = lambda x: region_imgs[x].shape[1])
            is_quadrilaterals = True
        
        texts = {}
        if config.use_mocr_merge:
            merged_textlines, merged_idx = await merge_bboxes(textlines, image.shape[1], image.shape[0])
            merged_quadrilaterals = list(self._generate_text_direction(merged_textlines))
        else:
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
                # Use pytesseract for OCR
                texts[idx] = pytesseract.image_to_string(Image.fromarray(merged_region_imgs[idx]), lang='eng')
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error during OCR: {e}")
                texts[idx] = ""
            
        ix = 0
        out_regions = {}
        for indices in chunks(perm, max_chunk_size):
            N = len(indices)
            widths = [region_imgs[i].shape[1] for i in indices]
            max_width = 4 * (max(widths) + 7) // 4
            region = np.zeros((N, text_height, max_width, 3), dtype = np.uint8)
            idx_keys = []
            for i, idx in enumerate(indices):
                idx_keys.append(idx)
                W = region_imgs[idx].shape[1]
                tmp = region_imgs[idx]
                region[i, :, : W, :]=tmp
                if verbose:
                    os.makedirs('result/ocrs/', exist_ok=True)
                    if quadrilaterals[idx][1] == 'v':
                        cv2.imwrite(f'result/ocrs/{ix}.png', cv2.rotate(cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE))
                    else:
                        cv2.imwrite(f'result/ocrs/{ix}.png', cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR))
                ix += 1
            # Populate out_regions with OCR results
            for i in range(N):
                cur_region = quadrilaterals[indices[i]][0]
                cur_region.text = texts[indices[i]]
                cur_region.prob = 1.0  # Set a default probability
                cur_region.area = cur_region.aabb.area()  # Ensure area is set
                out_regions[idx_keys[i]] = cur_region
                
        output_regions = []
        for i, nodes in enumerate(merged_idx):
            total_logprobs = 0
            total_area = 0
            fg_r = []
            fg_g = []
            fg_b = []
            bg_r = []
            bg_g = []
            bg_b = []
            
            for idx in nodes:
                if idx not in out_regions:
                    continue
                    
                total_logprobs += np.log(out_regions[idx].prob) * out_regions[idx].area
                total_area += out_regions[idx].area
                fg_r.append(out_regions[idx].fg_r)
                fg_g.append(out_regions[idx].fg_g)
                fg_b.append(out_regions[idx].fg_b)
                bg_r.append(out_regions[idx].bg_r)
                bg_g.append(out_regions[idx].bg_g)
                bg_b.append(out_regions[idx].bg_b)
                
            total_logprobs /= total_area
            prob = np.exp(total_logprobs)
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
            else: # TextBlock
                cur_region.text.append(txt)
                cur_region.update_font_colors(np.array([fr, fg, fb]), np.array([br, bg, bb]))
            output_regions.append(cur_region)

        if is_quadrilaterals:
            return output_regions
        return textlines
