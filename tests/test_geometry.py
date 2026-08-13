"""Verify the coordinate-tracking math that OCR box mapping relies on.

The critical invariant: a point mapped original -> processed (via the composed
preprocessing affine) and back through the inverse must return to itself. If this
breaks, every OCR box lands in the wrong place.
"""
import cv2
import numpy as np

from app.utils.geometry import (apply_affine_to_points, compose_affine,
                                identity_affine, invert_affine,
                                merge_boxes, polygon_to_bbox)


def test_identity_roundtrip():
    pts = [[10, 20], [300, 400]]
    m = identity_affine()
    out = apply_affine_to_points(pts, m)
    assert np.allclose(out, pts)


def test_scale_then_rotate_roundtrip():
    # Compose a scale (x1.4) followed by a 7-degree rotation, then invert.
    scale = np.array([[1.4, 0, 0], [0, 1.4, 0]], dtype=np.float64)
    rot = cv2.getRotationMatrix2D((500, 300), 7.0, 1.0)
    forward = compose_affine(scale, rot)          # original -> processed
    inverse = invert_affine(forward)              # processed -> original

    pts = [[12, 34], [656, 480], [700, 400], [0, 0]]
    processed = apply_affine_to_points(pts, forward)
    recovered = apply_affine_to_points(processed.tolist(), inverse)
    assert np.allclose(recovered, pts, atol=1e-6)


def test_polygon_to_bbox_clips():
    poly = [[-5, 10], [120, 12], [118, 60], [-3, 58]]
    bbox = polygon_to_bbox(poly, clip_w=100, clip_h=100)
    assert bbox == [0, 10, 100, 60]


def test_merge_boxes():
    boxes = [[100, 200, 180, 235], [185, 200, 270, 235], [275, 200, 370, 235]]
    assert merge_boxes(boxes) == [100, 200, 370, 235]
