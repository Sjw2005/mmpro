import copy
import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv


class DetectDC(Detect):
    """分类分支使用标准Conv的检测头，增强细粒度类别判别能力。"""

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        if self.end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)
