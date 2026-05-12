"""Compatibility blocks required by the YOLO-MP checkpoint.

The released checkpoint references these classes from
``ultralytics.nn.extra_modules.block``, but the example folder does not ship
that module. The saved checkpoint restores the child layers from the pickle;
these classes provide the matching forward methods for inference.
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import HGBlock


class Ghost_HGBlock(HGBlock):
    """HGBlock variant used by the YOLO-MP checkpoint."""


class FeaturePyramidSharedConv(nn.Module):
    """Feature pyramid block with one shared convolution applied at several scales."""

    def forward(self, x):
        y = self.cv1(x) if hasattr(self, "cv1") else x

        if hasattr(self, "m"):
            features = [m(y) for m in self.m]
        elif hasattr(self, "share_conv"):
            features = [
                y,
                self.share_conv(y),
                self.share_conv(y),
                self.share_conv(y),
            ]
        else:
            features = [y]

        y = torch.cat(features, 1) if len(features) > 1 else features[0]
        return self.cv2(y) if hasattr(self, "cv2") else y


class RGCSPELAN(nn.Module):
    """RepGhost CSP-ELAN block used by YOLO-MP."""

    def forward(self, x):
        y1, y2 = self.cv1(x).chunk(2, 1)
        features = [y1]

        for block in (self.cv3, self.cv4):
            y2 = block(y2)
            features.append(y2)

        return self.cv2(torch.cat(features, 1))
