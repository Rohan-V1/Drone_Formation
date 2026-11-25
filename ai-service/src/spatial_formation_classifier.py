"""Interpretable shape classifier for unordered 2D image-center points.

The classifier is deliberately conservative. It recognizes canonical V,
four-corner diamond/box, ring/circle, and multi-row geometry. Collinear arrays
receive image-axis layout labels only; a still image cannot establish flight
heading. When a set fits both a canonical shape and the multi-row heuristic,
the stronger spatial fit takes precedence. This is a geometric prototype,
not a trained or calibrated classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import pi

import numpy as np


FORMATION_CLASSES = (
    "V formation", "Inverted V formation", "Line/Abreast formation",
    "Column/File formation", "Echelon (Right, image-space) formation",
    "Echelon (Left, image-space) formation",
    "Diamond/Rhombus formation", "Box/Square formation", "Wedge formation",
    "Circle/Ring formation", "Line Formation (Multi-row)",
    "Dispersed/Swarm (no fixed pattern)",
)


@dataclass
class Classification:
    label: str
    quality: float
    num_positions: int
    rejection_reason: str
    candidate_scores: dict[str, float]


def _as_points(points):
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return np.empty((0, 2), dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("points must be a finite N x 2 array")
    # Exact duplicate centers do not add geometric evidence.
    return np.unique(p, axis=0)


def _normalise(p):
    centered = p - p.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if scale < 1e-9:
        return None, None
    return centered / scale, scale


def _v_error(p):
    """Best two-ray fit from any vertex; angular residual, range [0,1]."""
    n = len(p)
    if n < 5:
        return 1.0
    best = 1.0
    for apex_idx in range(n):
        vectors = p - p[apex_idx]
        vectors = vectors[np.arange(n) != apex_idx]
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > 1e-8
        vectors, norms = vectors[valid], norms[valid]
        if len(vectors) < 3:
            continue
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        for i, j in combinations(range(len(angles)), 2):
            a1, a2 = angles[i], angles[j]
            sep = abs((a1 - a2 + pi) % (2*pi) - pi)
            # A V needs a real opening angle and cannot be almost a straight line.
            if sep < np.deg2rad(28) or sep > np.deg2rad(155):
                continue
            d1 = np.abs((angles - a1 + pi) % (2*pi) - pi)
            d2 = np.abs((angles - a2 + pi) % (2*pi) - pi)
            arm = np.minimum(d1, d2)
            assignment = d1 <= d2
            if assignment.all() or (~assignment).all():
                continue
            # Angular residual is scale/rotation invariant. Penalize unbalanced
            # assignments slightly; drone counts may differ between arms.
            angular = float(np.mean(np.sin(np.minimum(arm, pi/2))))
            imbalance = abs(int(assignment.sum()) - int((~assignment).sum())) / len(assignment)
            best = min(best, angular + 0.025 * imbalance)
    return best


def _circle_error(p):
    if len(p) < 5:
        return 1.0
    centered = p - p.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[-1] <= 1e-10 or eigenvalues[0] / eigenvalues[-1] < 0.08:
        return 1.0

    # A circular group viewed obliquely can project to an ellipse. Whiten the
    # two image axes before measuring ring radii and angular gaps. The minimum
    # axis-ratio guard above prevents nearly collinear points becoming a ring.
    whitening = eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-10))) @ eigenvectors.T
    ring_points = centered @ whitening
    radii = np.linalg.norm(ring_points, axis=1)
    mean_r = float(radii.mean())
    if mean_r < 1e-8:
        return 1.0
    radial_cv = float(radii.std() / mean_r)
    angles = np.sort(np.arctan2(ring_points[:, 1], ring_points[:, 0]))
    gaps = np.diff(np.r_[angles, angles[0] + 2*pi])
    angular_cv = float(gaps.std() / max(gaps.mean(), 1e-8))
    return 0.65 * radial_cv + 0.35 * angular_cv


def _diamond_error(p):
    # The canonical box/diamond has four corners; a fifth center drone is
    # supported when it lies near the four-corner center.
    if len(p) not in (4, 5):
        return 1.0
    if len(p) == 5:
        center_idx = int(np.argmin(np.linalg.norm(p - p.mean(axis=0), axis=1)))
        if np.linalg.norm(p[center_idx] - p.mean(axis=0)) > 0.18:
            return 1.0
        p = np.delete(p, center_idx, axis=0)
    center = p.mean(axis=0)
    vec = p - center
    radii = np.linalg.norm(vec, axis=1)
    if radii.mean() < 1e-8:
        return 1.0
    angles = np.sort(np.arctan2(vec[:, 1], vec[:, 0]))
    gaps = np.diff(np.r_[angles, angles[0] + 2*pi])
    radial_cv = float(radii.std() / radii.mean())
    gap_error = float(np.sqrt(np.mean((gaps - pi/2)**2)) / (pi/2))
    return 0.55 * radial_cv + 0.45 * gap_error


def classify(points) -> Classification:
    """Classify an unordered set of image centers.

    `quality` is a residual-and-margin heuristic in [0,1], not a probability.
    Positions should be in a common frame and may optionally be normalized by
    image width/height before calling. Translation, uniform scale, and rotation
    are removed by centering and rotation-invariant geometric features.
    """
    p = _as_points(points)
    n = len(p)
    if n < 4:
        return Classification("Unknown", 0.0, n, "sparse: need at least 4 distinct positions", {})
    q, _ = _normalise(p)
    if q is None:
        return Classification("Unknown", 0.0, n, "positions are coincident", {})
    cov = np.cov(q.T)
    eigen = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0))[::-1]
    linear_ratio = float(eigen[1] / max(eigen[0], 1e-12))
    if linear_ratio < 0.025:
        return Classification(
            "Unknown", 0.0, n,
            "collinear layout is ambiguous between echelon, column/file, and line/abreast without heading",
            {"linear_ratio": linear_ratio},
        )

    limits = {"V formation": 0.16, "Diamond/Box formation": 0.22, "Circle formation": 0.24}
    errors = {
        "V formation": _v_error(q),
        "Diamond/Box formation": _diamond_error(q),
        "Circle formation": _circle_error(q),
    }
    qualities = {label: max(0.0, 1.0 - errors[label] / limits[label]) for label in errors}
    candidates = sorted(qualities.items(), key=lambda item: item[1], reverse=True)
    best_label, best_quality = candidates[0]
    second_quality = candidates[1][1]
    if best_quality < 0.55:
        return Classification("Unknown", best_quality, n, "no canonical template fits closely enough", qualities)
    if best_quality - second_quality < 0.12:
        return Classification("Unknown", best_quality, n, "ambiguous: competing geometric templates have similar scores", qualities)
    return Classification(best_label, best_quality, n, "", qualities)


def classify_image(points) -> Classification:
    """Classify image-space layouts for the 12-label reference chart.

    Directional labels describe only the arrangement in the image. They do
    not claim flight heading; video temporal analysis remains responsible for
    travel-relative Echelon/Column/Line assessments.
    """
    p = _as_points(points)
    if len(p) < 4:
        return Classification("Unknown", 0.0, len(p), "sparse: need at least 4 distinct positions", {})

    centered = p - p.mean(axis=0, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    ratio = float(max(0.0, min(eigenvalues)) / max(float(max(eigenvalues)), 1e-12))
    angle = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    line_limit = 0.025
    if ratio < line_limit:
        if angle > 90:
            angle = 180 - angle
        if angle <= 18:
            label = "Line/Abreast (image-horizontal)"
        elif angle >= 72:
            label = "Column/File (image-vertical)"
        else:
            side = "Right" if major[0] * major[1] < 0 else "Left"
            label = f"Echelon ({side}, image-space)"
        return Classification(label, max(0.0, 1.0 - ratio / line_limit), len(p),
                              "image-axis orientation only; a still image cannot establish travel direction",
                              {"line_ratio": ratio, "image_axis_angle_deg": round(angle, 3)})

    base = classify(p)
    if base.label == "V formation":
        v_error, apex, opening = _best_v_fit(p)
        if apex is not None and opening is not None:
            if opening >= 115:
                label = "Wedge formation"
            elif apex[1] < float(np.median(p[:, 1])):
                label = "Inverted V formation"
            else:
                label = "V formation"
            return Classification(label, base.quality, len(p),
                                  "image-space shape candidate; rotation, occlusion, and missing drones can change this assessment",
                                  {**base.candidate_scores, "v_fit_error": v_error,
                                   "v_opening_angle_deg": opening})
    if base.label == "Diamond/Box formation":
        edge_angle = _quadrilateral_edge_angle(p)
        if edge_angle is not None:
            if edge_angle <= 18:
                label = "Box/Square formation"
            elif edge_angle >= 27:
                label = "Diamond/Rhombus formation"
            else:
                return Classification("Unknown", base.quality, len(p),
                                      "ambiguous orientation between Box/Square and Diamond/Rhombus",
                                      {**base.candidate_scores, "edge_angle_deg": edge_angle})
            return Classification(label, base.quality, len(p),
                                  "image-space quadrilateral candidate; perspective may change its apparent orientation",
                                  {**base.candidate_scores, "edge_angle_deg": edge_angle})
    if base.label == "Circle formation":
        return Classification("Circle/Ring formation", base.quality, len(p),
                              "image-space ring candidate; projection and missed detections may affect the fit",
                              base.candidate_scores)

    # Evaluate canonical shapes before the multi-row heuristic. A V can form
    # several bands when its arms are projected onto the minor PCA axis, so
    # treating row count first systematically stole clear V cases.
    # Keep multi-row only when its fit is strong and beats a competing
    # canonical template by a clear margin.
    multi = _multirow_line(p)
    if multi is not None:
        quality, rows = multi
        if quality >= 0.65 and (base.label == "Unknown" or quality >= base.quality + 0.12):
            return Classification("Line Formation (Multi-row)", quality, len(p),
                                  f"{rows} repeated image-space rows were detected; this does not establish flight direction",
                                  {"multirow_quality": quality, "row_count": rows})
    if base.label != "Unknown":
        return base
    return base


def _multirow_line(points):
    """Find separated parallel bands with multiple points spanning each band."""
    p = points - points.mean(axis=0, keepdims=True)
    values, vectors = np.linalg.eigh(np.cov(p.T))
    order = np.argsort(values)[::-1]
    major_axis, minor_axis = vectors[:, order[0]], vectors[:, order[1]]
    major_angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])) % 180)
    if major_angle > 90:
        major_angle = 180 - major_angle
    if major_angle > 22 or float(values[order[1]] / max(float(values[order[0]]), 1e-12)) > 0.35:
        return None
    along, across = p @ major_axis, p @ minor_axis
    span = float(np.ptp(along))
    across_span = float(np.ptp(across))
    if span < 1e-6 or across_span < 0.018:
        return None
    sorted_indices = np.argsort(across)
    threshold = max(0.012, 0.22 * across_span)
    groups, group = [], [int(sorted_indices[0])]
    for previous, current in zip(sorted_indices[:-1], sorted_indices[1:]):
        if across[current] - across[previous] > threshold:
            groups.append(group)
            group = []
        group.append(int(current))
    groups.append(group)
    usable = [g for g in groups if len(g) >= 2 and np.ptp(along[g]) >= 0.22 * span]
    if not 2 <= len(usable) <= 5:
        return None
    row_centers = sorted(float(np.mean(across[g])) for g in usable)
    gaps = np.diff(row_centers)
    if len(gaps) > 1 and float(np.std(gaps) / max(np.mean(gaps), 1e-9)) > 0.85:
        return None
    coverage = min(float(np.ptp(along[g])) / span for g in usable)
    quality = float(np.clip(0.55 * coverage + 0.45 * (1 - min(1.0, np.std(gaps) / max(np.mean(gaps), 1e-9))), 0, 1))
    return quality, len(usable)


def _best_v_fit(points):
    best = (1.0, None, None)
    n = len(points)
    for apex_idx in range(n):
        vectors = points - points[apex_idx]
        vectors = vectors[np.arange(n) != apex_idx]
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > 1e-8
        vectors, norms = vectors[valid], norms[valid]
        if len(vectors) < 3:
            continue
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        for i, j in combinations(range(len(angles)), 2):
            separation = abs((angles[i] - angles[j] + pi) % (2 * pi) - pi)
            if separation < np.deg2rad(28) or separation > np.deg2rad(155):
                continue
            d1 = np.abs((angles - angles[i] + pi) % (2 * pi) - pi)
            d2 = np.abs((angles - angles[j] + pi) % (2 * pi) - pi)
            assignment = d1 <= d2
            if assignment.all() or (~assignment).all():
                continue
            residual = float(np.mean(np.sin(np.minimum(np.minimum(d1, d2), pi / 2))))
            if residual < best[0]:
                best = (residual, points[apex_idx], float(np.degrees(separation)))
    return best


def _quadrilateral_edge_angle(points):
    p = np.asarray(points, dtype=float)
    if len(p) == 5:
        center = p.mean(axis=0)
        p = np.delete(p, int(np.argmin(np.linalg.norm(p - center, axis=1))), axis=0)
    if len(p) != 4:
        return None
    center = p.mean(axis=0)
    order = np.argsort(np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0]))
    ordered = p[order]
    angles = []
    for i in range(4):
        delta = ordered[(i + 1) % 4] - ordered[i]
        if np.linalg.norm(delta) < 1e-9:
            return None
        angle = abs(float(np.degrees(np.arctan2(delta[1], delta[0])))) % 90
        angles.append(min(angle, 90 - angle))
    return round(float(np.mean(angles)), 3)


def classify_video_window(frames, camera_affines, image_width, image_height, sample_period=0.1):
    """Conservatively assess real tracked positions in one short video window.

    `frames` is a sequence of per-frame dictionaries with tracked IDs and
    original-pixel x/y centers. `camera_affines[i]` maps the preceding sampled
    frame to frame i and includes a boolean `reliable` flag. This rule-based
    path uses no learned formation model. `quality` values are heuristics, not
    probabilities.
    """
    n_frames = len(frames)
    if n_frames < 10 or image_width <= 0 or image_height <= 0:
        return _window_result("Unknown", "too few valid sampled frames", 0.0, 0, 0.0)

    frame_maps = []
    track_observations = {}
    for frame_index, frame in enumerate(frames):
        points = {}
        for row in frame:
            track_id = row.get("track_id")
            if track_id is None or not np.isfinite([row.get("x", np.nan), row.get("y", np.nan)]).all():
                continue
            points[str(track_id)] = np.array([float(row["x"]) / image_width,
                                              float(row["y"]) / image_height], dtype=float)
            track_observations.setdefault(str(track_id), []).append((frame_index, points[str(track_id)]))
        frame_maps.append(points)

    min_observations = int(np.ceil(0.70 * n_frames))
    persistent_ids = sorted(tid for tid, obs in track_observations.items()
                            if len(obs) >= min_observations)
    simultaneous = sum(sum(tid in frame for tid in persistent_ids) >= 4 for frame in frame_maps)
    if len(persistent_ids) < 4:
        return _window_result("Unknown", "fewer than four persistent tracked drones", 0.0,
                              len(persistent_ids), 0.0, simultaneous_fraction=simultaneous / n_frames)
    if simultaneous / n_frames < 0.70:
        return _window_result("Unknown", "too few simultaneous persistent tracks", 0.0,
                              len(persistent_ids), 0.0, simultaneous_fraction=simultaneous / n_frames)

    # Assess spatial shapes over multiple frames. Direction-dependent line
    # labels are deliberately excluded here and must pass the heading gate.
    geometry_only_labels = {
        "V formation", "Inverted V formation", "Wedge formation",
        "Diamond/Rhombus formation", "Box/Square formation",
        "Circle/Ring formation", "Line Formation (Multi-row)",
    }
    shape_counts = {}
    shape_qualities = {}
    for frame in frame_maps:
        pts = [frame[tid] for tid in persistent_ids if tid in frame]
        if len(pts) < 4:
            continue
        result = classify_image(pts)
        if result.label in geometry_only_labels:
            shape_counts[result.label] = shape_counts.get(result.label, 0) + 1
            shape_qualities.setdefault(result.label, []).append(float(result.quality))
    evaluated_frames = sum(1 for frame in frame_maps
                           if sum(tid in frame for tid in persistent_ids) >= 4)
    stable_error, stable_frames = _relative_geometry_error(frame_maps, persistent_ids)
    if shape_counts and evaluated_frames:
        best_label, votes = max(shape_counts.items(), key=lambda item: item[1])
        agreement = votes / evaluated_frames
        mean_quality = float(np.mean(shape_qualities[best_label]))
        if agreement >= 0.70 and stable_frames >= 0.70 * evaluated_frames and stable_error <= 0.28:
            return _window_result(best_label.removesuffix(" formation"),
                                  "spatial pattern was consistent across tracked video frames",
                                  mean_quality * agreement, len(persistent_ids), stable_error,
                                  geometry_agreement=agreement, heading_reliable=False,
                                  spatial_class=best_label)

    # Direction-dependent line classes need camera-compensated motion. If any
    # sampled interval lacks a reliable background transform, abstain.
    if camera_affines is None or len(camera_affines) != n_frames:
        return _window_result("Unknown", "reliable camera-motion estimates are unavailable for directional classes",
                              0.0, len(persistent_ids), stable_error, heading_reliable=False)
    if any(not item or not item.get("reliable") or item.get("matrix") is None
           for item in camera_affines[1:]):
        return _window_result("Unknown", "background motion could not be estimated reliably; travel direction is uncertain",
                              0.0, len(persistent_ids), stable_error, heading_reliable=False)

    cumulative = [np.eye(3, dtype=float)]
    for affine in camera_affines[1:]:
        matrix = np.eye(3, dtype=float)
        matrix[:2, :] = np.asarray(affine["matrix"], dtype=float)
        cumulative.append(matrix @ cumulative[-1])
    stabilized = []
    for frame_index, frame in enumerate(frame_maps):
        inverse = np.linalg.inv(cumulative[frame_index])
        mapped = {}
        for tid, point in frame.items():
            original_pixel = np.array([point[0] * image_width, point[1] * image_height, 1.0])
            ref_pixel = inverse @ original_pixel
            mapped[tid] = np.array([ref_pixel[0] / image_width, ref_pixel[1] / image_height])
        stabilized.append(mapped)

    paths = {}
    for tid in persistent_ids:
        observations = [(i, frame[tid]) for i, frame in enumerate(stabilized) if tid in frame]
        if len(observations) < min_observations or any(
            observations[k + 1][0] - observations[k][0] > 2 for k in range(len(observations) - 1)
        ):
            continue
        times = np.asarray([i * sample_period for i, _ in observations], dtype=float)
        positions = np.asarray([p for _, p in observations], dtype=float)
        vx = float(np.polyfit(times, positions[:, 0], 1)[0])
        vy = float(np.polyfit(times, positions[:, 1], 1)[0])
        paths[tid] = {"mean": positions.mean(axis=0), "velocity": np.array([vx, vy]),
                      "positions": positions}
    if len(paths) < 4:
        return _window_result("Unknown", "track continuity is insufficient to estimate shared travel direction",
                              0.0, len(paths), stable_error, heading_reliable=False)

    velocities = np.asarray([v["velocity"] for v in paths.values()])
    speeds = np.linalg.norm(velocities, axis=1)
    moving = speeds > 1e-6
    if moving.sum() < 4:
        return _window_result("Unknown", "tracks show too little movement to determine travel direction",
                              0.0, len(paths), stable_error, heading_reliable=False)
    velocities, speeds = velocities[moving], speeds[moving]
    directions = velocities / speeds[:, None]
    heading = np.median(velocities, axis=0)
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm <= 1e-8:
        return _window_result("Unknown", "track motion directions conflict", 0.0,
                              len(paths), stable_error, heading_reliable=False)
    heading_unit = heading / heading_norm
    coherence = float(np.clip(np.mean(directions @ heading_unit), 0.0, 1.0))
    duration = max((n_frames - 1) * sample_period, sample_period)
    centers = np.asarray([v["mean"] for v in paths.values()])
    centered = centers - centers.mean(axis=0)
    covariance = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, 0)
    linear_ratio = float(eigvals[0] / max(eigvals[-1], 1e-12))
    if linear_ratio > 0.12:
        return _window_result("Unknown", "group geometry is not sufficiently line-shaped for directional classes",
                              0.0, len(paths), stable_error, heading_reliable=coherence >= 0.75,
                              heading_coherence=coherence, line_ratio=linear_ratio)
    if stable_frames < 0.70 * evaluated_frames or stable_error > 0.28:
        return _window_result("Unknown", "relative positions changed too much during the window",
                              0.0, len(paths), stable_error, heading_reliable=coherence >= 0.75,
                              heading_coherence=coherence)
    if coherence < 0.75:
        return _window_result("Unknown", "drone motion does not share a reliable direction",
                              0.0, len(paths), stable_error, heading_reliable=False,
                              heading_coherence=coherence)
    formation_scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    displacement_ratio = heading_norm * duration / max(formation_scale, 1e-9)
    if displacement_ratio < 0.20:
        return _window_result("Unknown", "camera-compensated travel is too small relative to formation size",
                              0.0, len(paths), stable_error, heading_reliable=True,
                              heading_coherence=coherence, displacement_ratio=displacement_ratio)

    formation_axis = eigvecs[:, -1]
    angle = float(np.degrees(np.arccos(np.clip(abs(float(formation_axis @ heading_unit)), 0.0, 1.0))))
    if angle <= 25.0:
        label = "Column/File"
    elif angle >= 65.0:
        label = "Line/Abreast"
    else:
        label = "Echelon"
    quality = float(np.clip(0.40 * coherence + 0.35 * (1.0 - stable_error / 0.28)
                            + 0.25 * (1.0 - linear_ratio / 0.12), 0.0, 1.0))
    return _window_result(label, "class inferred from stable line geometry relative to camera-compensated travel direction",
                          quality, len(paths), stable_error, heading_reliable=True,
                          heading_coherence=coherence, direction_angle_deg=angle,
                          displacement_ratio=displacement_ratio, line_ratio=linear_ratio)


def _relative_geometry_error(frame_maps, track_ids):
    usable = [frame for frame in frame_maps if sum(tid in frame for tid in track_ids) >= 4]
    if len(usable) < 2:
        return 1.0, len(usable)
    common = [tid for tid in track_ids if sum(tid in frame for frame in usable) >= int(np.ceil(0.8 * len(usable)))]
    if len(common) < 4:
        return 1.0, 0
    vectors = []
    for frame in usable:
        pts = np.asarray([frame[tid] for tid in common if tid in frame])
        if len(pts) < 4:
            continue
        d = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d.append(float(np.linalg.norm(pts[i] - pts[j])))
        scale = float(np.mean(d)) if d else 0.0
        if scale > 1e-9:
            vectors.append(np.asarray(d) / scale)
    if len(vectors) < 0.7 * len(usable) or not vectors:
        return 1.0, len(vectors)
    matrix = np.vstack(vectors)
    return float(np.sqrt(np.mean(np.var(matrix, axis=0)))), len(vectors)


def _window_result(label, reason, quality, usable_tracks, stability_error, **evidence):
    return {"prediction": label, "reason": reason, "geometry_quality": round(float(quality), 6),
            "usable_tracks": int(usable_tracks), "geometry_stability_error": round(float(stability_error), 6),
            "heading_reliable": bool(evidence.pop("heading_reliable", False)), **evidence}


