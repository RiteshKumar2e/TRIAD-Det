"""TRIAD-Det: Triple-branch Risk-Aware Inspection Detector.

A single-stage, anchor-free detector written directly in PyTorch. It shares no code or
building blocks with the YOLO family (no C2f / C3k2 / SPPF / C2PSA / DFL, no Ultralytics).

    backbone   generic timm CNN (default ConvNeXt-Tiny), 4 feature levels C2..C5
    neck       bidirectional fusion: top-down FPN pass, then a bottom-up pass, GroupNorm
    branches   1. detection: per-location class logits + box distances (l, t, r, b)
               2. quality:   per-location IoU-aware localisation quality
               3. census:    image-level defect count (Poisson rate) and class presence

The census branch exists for the risk-control study: at inference the gap between the
expected number of defects and the number of confident boxes is a ground-truth-free signal
that a defect may have been missed.
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms, generalized_box_iou_loss, sigmoid_focal_loss


def conv_gn(cin, cout, k=3, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.GroupNorm(16, cout), nn.SiLU(inplace=True))


class BiFusionNeck(nn.Module):
    """Top-down then bottom-up fusion over C3..C5, plus an extra stride-64 level."""

    def __init__(self, in_channels, ch):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, ch, 1) for c in in_channels)
        self.td = nn.ModuleList(conv_gn(ch, ch) for _ in in_channels)
        self.down = nn.ModuleList(conv_gn(ch, ch, 3, 2) for _ in in_channels[1:])
        self.bu = nn.ModuleList(conv_gn(2 * ch, ch) for _ in in_channels[1:])
        self.extra = conv_gn(ch, ch, 3, 2)

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        td = [None] * len(lat)
        td[-1] = self.td[-1](lat[-1])
        for i in range(len(lat) - 2, -1, -1):
            up = F.interpolate(td[i + 1], size=lat[i].shape[-2:], mode="nearest")
            td[i] = self.td[i](lat[i] + up)
        out = [td[0]]
        for i in range(1, len(td)):
            out.append(self.bu[i - 1](torch.cat([td[i], self.down[i - 1](out[-1])], 1)))
        out.append(self.extra(out[-1]))
        return out


class Head(nn.Module):
    def __init__(self, ch, num_classes, convs, n_levels):
        super().__init__()
        self.cls_tower = nn.Sequential(*[conv_gn(ch, ch) for _ in range(convs)])
        self.box_tower = nn.Sequential(*[conv_gn(ch, ch) for _ in range(convs)])
        self.cls = nn.Conv2d(ch, num_classes, 3, padding=1)
        self.box = nn.Conv2d(ch, 4, 3, padding=1)
        self.quality = nn.Conv2d(ch, 1, 3, padding=1)
        self.scales = nn.Parameter(torch.ones(n_levels))
        nn.init.constant_(self.cls.bias, -math.log(99))
        for layer in (self.cls, self.box, self.quality):
            nn.init.normal_(layer.weight, std=0.01)
        nn.init.zeros_(self.box.bias)
        nn.init.zeros_(self.quality.bias)

    def forward(self, feats, strides):
        cls, box, qual = [], [], []
        for i, (f, s) in enumerate(zip(feats, strides)):
            c, b = self.cls_tower(f), self.box_tower(f)
            cls.append(self.cls(c).flatten(2).transpose(1, 2))
            box.append((F.relu(self.box(b) * self.scales[i]) * s).flatten(2).transpose(1, 2))
            qual.append(self.quality(b).flatten(2).transpose(1, 2))
        return torch.cat(cls, 1), torch.cat(box, 1), torch.cat(qual, 1).squeeze(-1)


class CensusHead(nn.Module):
    """Image-level branch: log defect count and multi-label class presence."""

    def __init__(self, ch, num_classes, n_levels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ch * n_levels, 256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 1 + num_classes))

    def forward(self, feats):
        pooled = torch.cat([f.mean((2, 3)) for f in feats], 1)
        out = self.mlp(pooled)
        return out[:, 0], out[:, 1:]


class TriadDet(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.num_classes = num_classes
        self.strides = cfg["strides"]
        self.ranges = cfg["size_ranges"]
        self.radius = cfg["center_radius"]
        self.backbone = timm.create_model(cfg["backbone"], pretrained=cfg["pretrained_backbone"], features_only=True,
                                          out_indices=(1, 2, 3))
        chans = self.backbone.feature_info.channels()
        ch = cfg["neck_channels"]
        self.neck = BiFusionNeck(chans, ch)
        self.head = Head(ch, num_classes, cfg["head_convs"], len(self.strides))
        self.census = CensusHead(ch, num_classes, len(self.strides))
        self._points = {}

    def forward(self, images):
        feats = self.neck(self.backbone(images))
        cls, box, qual = self.head(feats, self.strides)
        log_count, presence = self.census(feats)
        return {"cls": cls, "box": box, "quality": qual, "log_count": log_count, "presence": presence,
                "shapes": [f.shape[-2:] for f in feats]}

    # ------------------------------------------------------------------ geometry
    def points(self, shapes, device):
        key = (tuple(tuple(s) for s in shapes), str(device))
        if key not in self._points:
            pts, strides, lo, hi = [], [], [], []
            for (h, w), s, (a, b) in zip(shapes, self.strides, self.ranges):
                ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
                p = torch.stack([(xs.flatten() + 0.5) * s, (ys.flatten() + 0.5) * s], 1).float()
                pts.append(p)
                n = len(p)
                strides.append(torch.full((n,), float(s), device=device))
                lo.append(torch.full((n,), float(a), device=device))
                hi.append(torch.full((n,), float(b), device=device))
            self._points[key] = tuple(torch.cat(x) for x in (pts, strides, lo, hi))
        return self._points[key]

    @staticmethod
    def decode(points, dist):
        return torch.cat([points - dist[..., :2], points + dist[..., 2:]], -1)

    # ------------------------------------------------------------------ training
    @torch.no_grad()
    def assign(self, points, strides, lo, hi, gt_boxes, gt_labels):
        """FCOS-style assignment with centre sampling; ties go to the smallest box."""
        n = len(points)
        labels = torch.full((n,), -1, dtype=torch.long, device=points.device)
        targets = torch.zeros((n, 4), device=points.device)
        if len(gt_boxes) == 0:
            return labels, targets
        x, y = points[:, 0:1], points[:, 1:2]
        l, t = x - gt_boxes[:, 0], y - gt_boxes[:, 1]
        r, b = gt_boxes[:, 2] - x, gt_boxes[:, 3] - y
        ltrb = torch.stack([l, t, r, b], -1)                       # N x G x 4
        cx, cy = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2, (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        rad = strides[:, None] * self.radius
        in_centre = ((x - cx).abs() < rad) & ((y - cy).abs() < rad)
        in_box = ltrb.min(-1).values > 0
        extent = ltrb.max(-1).values
        in_range = (extent >= lo[:, None]) & (extent <= hi[:, None])
        area = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]))[None].expand(n, -1).clone()
        area[~(in_centre & in_box & in_range)] = float("inf")
        best_area, best = area.min(1)
        pos = torch.isfinite(best_area)
        labels[pos] = gt_labels[best[pos]]
        targets[pos] = gt_boxes[best[pos]]
        return labels, targets

    @torch.no_grad()
    def assign_atss(self, points, strides, gt_boxes, gt_labels, topk=9, anchor_scale=8.0):
        """ATSS (Zhang et al., CVPR 2020): per GT, the topk points nearest its centre on every level are
        candidates; positives are candidates whose anchor-box IoU >= mean + std of the candidates' IoUs
        and whose point lies inside the GT. A point claimed by several GTs goes to the highest IoU.
        No hand-set size ranges, so it adapts to each dataset's box sizes."""
        n = len(points)
        labels = torch.full((n,), -1, dtype=torch.long, device=points.device)
        targets = torch.zeros((n, 4), device=points.device)
        g = len(gt_boxes)
        if g == 0:
            return labels, targets
        half = strides[:, None] * anchor_scale / 2
        anchors = torch.cat([points - half, points + half], 1)
        ious = _pairwise_iou(anchors, gt_boxes)                                  # N x G
        centres = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        dist = torch.cdist(points, centres)                                      # N x G
        candidate = torch.zeros((n, g), dtype=torch.bool, device=points.device)
        start = 0
        for s in self.strides:
            count = int((strides == float(s)).sum())
            k = min(topk, count)
            idx = dist[start:start + count].topk(k, dim=0, largest=False).indices + start   # k x G
            candidate.scatter_(0, idx, True)
            start += count
        cand_iou = torch.where(candidate, ious, torch.zeros_like(ious))
        n_c = candidate.sum(0).clamp(min=1)
        mean = cand_iou.sum(0) / n_c
        std = (((cand_iou - mean) ** 2) * candidate).sum(0).div(n_c).sqrt()
        x, y = points[:, 0:1], points[:, 1:2]
        inside = (x > gt_boxes[:, 0]) & (x < gt_boxes[:, 2]) & (y > gt_boxes[:, 1]) & (y < gt_boxes[:, 3])
        positive = candidate & (ious >= mean + std) & inside
        empty = ~positive.any(0)                                                 # tiny GT: keep its best candidate
        if empty.any():
            best = cand_iou.argmax(0)
            cols = empty.nonzero().squeeze(1)
            positive[best[cols], cols] = True
        score = torch.where(positive, ious, torch.full_like(ious, -1.0))
        best_iou, best = score.max(1)
        pos = best_iou >= 0
        labels[pos] = gt_labels[best[pos]]
        targets[pos] = gt_boxes[best[pos]]
        return labels, targets

    def loss(self, out, gt_boxes, gt_labels, cfg):
        points, strides, lo, hi = self.points(out["shapes"], out["cls"].device)
        cls_t, box_t = [], []
        for boxes, labels in zip(gt_boxes, gt_labels):
            if cfg.get("assigner", "fcos") == "atss":
                lab, tgt = self.assign_atss(points, strides, boxes, labels)
            else:
                lab, tgt = self.assign(points, strides, lo, hi, boxes, labels)
            cls_t.append(lab)
            box_t.append(tgt)
        cls_t, box_t = torch.stack(cls_t), torch.stack(box_t)
        pos = cls_t >= 0
        n_pos = max(float(pos.sum()), 1.0)
        target = torch.zeros_like(out["cls"], dtype=torch.float32)
        pred_boxes = inter_iou = None
        if pos.any():
            pred_boxes = self.decode(points[None].expand(len(cls_t), -1, -1)[pos], out["box"][pos].float())
            with torch.no_grad():
                inter_iou = _paired_iou(pred_boxes.detach(), box_t[pos]).clamp(0, 1)
        if cfg.get("cls_loss", "focal") == "qfl" and pos.any():
            # Quality Focal Loss (Li et al., NeurIPS 2020): the positive class target is the IoU of the
            # predicted box, so the class score also ranks localisation quality.
            target[pos, cls_t[pos]] = inter_iou
            logits = out["cls"].float()
            prob = logits.sigmoid()
            loss_cls = (F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                        * (target - prob).abs().pow(2.0)).sum() / max(float(inter_iou.sum()), 1.0)
        else:
            target[pos, cls_t[pos]] = 1.0
            loss_cls = sigmoid_focal_loss(out["cls"].float(), target, alpha=0.25, gamma=2.0, reduction="sum") / n_pos
        if pos.any():
            iou_loss = generalized_box_iou_loss(pred_boxes, box_t[pos], reduction="none")
            loss_box = iou_loss.mean()
            loss_q = F.binary_cross_entropy_with_logits(out["quality"][pos].float(), inter_iou)
        else:
            loss_box = loss_q = out["box"].sum() * 0
        counts = torch.tensor([len(b) for b in gt_boxes], dtype=torch.float32, device=out["cls"].device)
        loss_count = F.poisson_nll_loss(out["log_count"].float(), counts, log_input=True)
        presence_t = torch.zeros_like(out["presence"])
        for i, labels in enumerate(gt_labels):
            presence_t[i, labels] = 1.0
        loss_pres = F.binary_cross_entropy_with_logits(out["presence"].float(), presence_t)
        total = loss_cls + 2.0 * loss_box + loss_q + cfg["count_loss_weight"] * loss_count + cfg["presence_loss_weight"] * loss_pres
        return total, {"cls": loss_cls.item(), "box": loss_box.item(), "quality": loss_q.item(),
                       "count": loss_count.item(), "presence": loss_pres.item()}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, out, conf, iou, max_det, topk):
        points, _, _, _ = self.points(out["shapes"], out["cls"].device)
        scores_all = (out["cls"].float().sigmoid() * out["quality"].float().sigmoid()[..., None]).sqrt()
        results = []
        for b in range(len(scores_all)):
            scores = scores_all[b].flatten()
            keep = scores > conf
            idx = keep.nonzero().squeeze(1)
            if len(idx) > topk:
                idx = idx[scores[idx].topk(topk).indices]
            loc, cls = idx // self.num_classes, idx % self.num_classes
            boxes = self.decode(points[loc], out["box"][b, loc].float())
            s = scores[idx]
            keep = batched_nms(boxes, s, cls, iou)[:max_det]
            results.append({"boxes": boxes[keep], "scores": s[keep], "labels": cls[keep],
                            "count": float(out["log_count"][b].float().exp()),
                            "presence": out["presence"][b].float().sigmoid()})
        return results


def _pairwise_iou(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None]
    return inter / (area_a + area_b - inter).clamp(min=1e-6)


def _paired_iou(a, b):
    lt = torch.max(a[:, :2], b[:, :2])
    rb = torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)
    return inter / (area(a) + area(b) - inter).clamp(min=1e-6)
