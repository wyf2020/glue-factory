Forked glue-factory to reproduce the AUC and timing results with and without Flash attention on various settings.

### install
Glue Factory runs with Python 3 and [PyTorch](https://pytorch.org/). The following installs the library and its basic dependencies:
```bash
git clone https://github.com/cvg/glue-factory
conda create -n glue python=3.8
cd glue-factory
python3 -m pip install -e .  # editable mode
# Please use RTX3090 to fully reproduce results.
# [2024.3.29]: Now, the newest version of torch is 2.2.2, pip install torch==2.2.1 to reproduce the results.
```

### SP+LG on ScanNet
```bash
# img_size=1296×968 set by gluefactory and max_num_keypoints/detection_threshold/nms_radius in gluefactory
python -m gluefactory.eval.scannet1500 --conf superpoint+lightglue-official_1296_2048_0_3 --overwrite # Lightglue without prune
python -m gluefactory.eval.scannet1500 --conf superpoint+lightglue-official_1296_2048_0_3 model.matcher.{depth_confidence=0.95,width_confidence=0.95} --overwrite # use prune confidence set by gluefactory
python -m gluefactory.eval.scannet1500 --conf superpoint+lightglue-official_1296_2048_0_3 model.matcher.{depth_confidence=0.95,width_confidence=0.99} --overwrite # use prune confidence in LightGlue official repository
# follow img_size=640×480 in SuperGlue paper's ScanNet setting and max_num_keypoints/detection_threshold/nms_radius in gluefactory
python -m gluefactory.eval.scannet1500 --conf superpoint+lightglue-official_640_2048_0_3 --overwrite # Lightglue without prune
python -m gluefactory.eval.scannet1500 --conf superpoint+lightglue-official_640_2048_0_3 model.matcher.{depth_confidence=0.95,width_confidence=0.95} --overwrite # use prune confidence set by gluefactory
python -m gluefactory.eval.scannet1500 --conf superpoint+lightglue-official_640_2048_0_3 model.matcher.{depth_confidence=0.95,width_confidence=0.99} --overwrite # use prune confidence in LightGlue official repository
# follow img_size=640×480 in SuperGlue paper's ScanNet setting and max_num_keypoints/detection_threshold/nms_radius in LightGlue official repository
python -m gluefactory.eval.scannet1500 --conf superpoint+lightglue-official_640_2048_5e-4_4 --overwrite # Lightglue without prune
python -m gluefactory.eval.scannet1500 --conf superpoint+lightglue-official_640_2048_5e-4_4 model.matcher.{depth_confidence=0.95,width_confidence=0.95} --overwrite # use prune confidence set by gluefactory
python -m gluefactory.eval.scannet1500 --conf superpoint+lightglue-official_640_2048_5e-4_4 model.matcher.{depth_confidence=0.95,width_confidence=0.99} --overwrite # use prune confidence in LightGlue official repository
```

### How we run LG without Flash-Attention
If you use official install guide, it will install the newest version of pytorch(2.2.1).

With torch >= 2.0.0, the code of SP+LG will automatically use Flash-Attention for better performance regardless of the `--flash` config of Lightglue.

Modifying the code in 'glue-factory/gluefactory/models/matchers/lightglue.py' will make no effect, because the 'glue-factory/gluefactory/models/matchers/lightglue_pretrained.py' will import the lightglue install by pip.

Therefore, to disable the Flash-Attention, you need to change the code in the lightglue pip package.

Suppose the conda environment name of glue-factory is `glue`, and python=3.8. You can find the code in '/path/to/miniconda3/envs/glue/lib/python3.8/site-packages/lightglue/lightglue.py' 
and set 'FLASH_AVAILABLE = False' in line 17, like this:
```python
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = False
else:
    FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True
```
You can use 'conda info' to find the path of the conda environment.
Or you can just install torch=1.13.1 to disable the Flash-Attention, but leads to a slightly different AUC results from torch=2.2.1.

### How we run LG with FP16 + Flash-Attention
1. In gluefactory/models/matchers/lightglue_pretrained.py, convert the model and input to half().
2. Find '/path/to/miniconda3/envs/glue/lib/python3.8/site-packages/lightglue/lightglue.py'.
3. comment out '@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)' in line 24.
4. Add dtype in Line 602-603:
```python
    mscores0_ = torch.zeros((b, m), device=mscores0.device, dtype=mscores0.dtype)
    mscores1_ = torch.zeros((b, n), device=mscores1.device, dtype=mscores1.dtype)
```
5. comment out autocast in Line 472:
```python
    # with torch.autocast(enabled=self.conf.mp, device_type="cuda"): # !4
    #     return self._forward(data)
    return self._forward(data)
```

### How we use RANSAC to evaluate the results
```python
    pred = {k: v[0].cpu() for k, v in pred.items()}
    data['view0'] = {k: v.cpu() for k, v in data['view0'].items()}
    data['view1'] = {k: v.cpu() for k, v in data['view1'].items()}
    data['T_1to0'] =  data['T_1to0'].cpu()
    data['T_0to1'] =  data['T_0to1'].cpu()
    pose_results_i = eval_relative_pose_robust(
        data,
        pred,
        {"estimator": 'opencv', "ransac_th": 0.5},
    )
```