"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

from omegaconf import OmegaConf

from . import get_model
from .base_model import BaseModel

to_ctr = OmegaConf.to_container  # convert DictConfig to dict

import torch
class TwoViewPipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
    }
    required_data_keys = ["view0", "view1"]
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "matcher",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )
        self.total_sp_ms = 0
        self.total_lg_ms = 0
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.warmup = False


    def extract_view(self, data, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        self.start_event.record()
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        if self.conf.extractor.name and not skip_extract:
            pred_i = {**pred_i, **self.extractor(data_i)}
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
        self.end_event.record()
        torch.cuda.synchronize()

        return pred_i

    def _forward(self, data):
        if not self.warmup:
            for _ in range(50):
                # pred0 = self.extract_view(data, "0")
                data_0 = data[f"view{0}"]
                pred0 = data_0.get("cache", {})
                skip_extract = len(pred0) > 0 and self.conf.allow_no_extract
                if self.conf.extractor.name and not skip_extract:
                    pred0 = {**pred0, **self.extractor(data_0)}
                elif self.conf.extractor.name and not self.conf.allow_no_extract:
                    pred0 = {**pred0, **self.extractor({**data_0, **pred0})}

                # pred1 = self.extract_view(data, "1")
                data_1 = data[f"view{1}"]
                pred1 = data_1.get("cache", {})
                skip_extract = len(pred1) > 0 and self.conf.allow_no_extract
                if self.conf.extractor.name and not skip_extract:
                    pred1 = {**pred1, **self.extractor(data_1)}
                elif self.conf.extractor.name and not self.conf.allow_no_extract:
                    pred1 = {**pred1, **self.extractor({**data_1, **pred1})}
                pred = {
                    **{k + "0": v for k, v in pred0.items()},
                    **{k + "1": v for k, v in pred1.items()},
                }

                pred = {**pred, **self.matcher({**data, **pred})}
            self.warmup=True

        pred0 = self.extract_view(data, "0")
        self.total_sp_ms += self.start_event.elapsed_time(self.end_event)
        pred1 = self.extract_view(data, "1")
        self.total_sp_ms += self.start_event.elapsed_time(self.end_event)
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        }

        self.start_event.record()
        if self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}
        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}
        self.end_event.record()
        torch.cuda.synchronize()
        self.total_lg_ms += self.start_event.elapsed_time(self.end_event)

        if self.conf.ground_truth.name and self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
        return pred

    def loss(self, pred, data):
        losses = {}
        metrics = {}
        total = 0

        # get labels
        if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        return {**losses, "total": total}, metrics
