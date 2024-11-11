from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import total_ordering
from itertools import product
from pathlib import Path
from typing import Any, Sequence, cast

import cv2 as cv
import numpy as np
from imagehash import ImageHash, phash
from PIL import Image
from shapely.affinity import scale
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon

from .scryfall import CLAHE, IMAGES_DIR

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# OpenCV config
MATCH_MODE = cv.NORM_HAMMING2
NFEATURES = 500
READ_MODE = cv.IMREAD_GRAYSCALE


@total_ordering
class Match:
    def __init__(self, gid: str, count: int) -> None:
        self.gid = gid
        self.count = count

    def __repr__(self) -> str:
        return f"Match<{self.gid}, {self.count}>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.gid == other
        if not isinstance(other, Match):
            return NotImplemented
        return self.gid == other.gid

    def __lt__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.gid < other
        if not isinstance(other, Match):
            return NotImplemented
        return self.gid < other.gid


def order_polygon_points(
    x: np.ndarray[Any, np.dtype[np.float64]],
    y: np.ndarray[Any, np.dtype[np.float64]],
):
    """
    Orders polygon points into a counterclockwise order.
    x_p, y_p are the x and y coordinates of the polygon points.
    """
    angle = np.arctan2(y - np.average(y), x - np.average(x))
    ind = np.argsort(angle)
    return (x[ind], y[ind])


def four_point_transform(image: cv.typing.MatLike, poly: Polygon):
    """
    A perspective transform for a quadrilateral polygon.
    Slightly modified version of the same function from
    https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
    """
    pts = np.zeros((4, 2))
    pts[:, 0] = np.asarray(poly.exterior.coords)[:-1, 0]
    pts[:, 1] = np.asarray(poly.exterior.coords)[:-1, 1]
    # obtain a consistent order of the points and unpack them
    # individually
    rect = np.zeros((4, 2))
    (rect[:, 0], rect[:, 1]) = order_polygon_points(pts[:, 0], pts[:, 1])

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    # width_a = np.sqrt(((b_r[0] - b_l[0]) ** 2) + ((b_r[1] - b_l[1]) ** 2))
    # width_b = np.sqrt(((t_r[0] - t_l[0]) ** 2) + ((t_r[1] - t_l[1]) ** 2))
    width_a = np.sqrt(((rect[1, 0] - rect[0, 0]) ** 2) + ((rect[1, 1] - rect[0, 1]) ** 2))
    width_b = np.sqrt(((rect[3, 0] - rect[2, 0]) ** 2) + ((rect[3, 1] - rect[2, 1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((rect[0, 0] - rect[3, 0]) ** 2) + ((rect[0, 1] - rect[3, 1]) ** 2))
    height_b = np.sqrt(((rect[1, 0] - rect[2, 0]) ** 2) + ((rect[1, 1] - rect[2, 1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order

    rect = np.array(
        [
            [rect[0, 0], rect[0, 1]],
            [rect[1, 0], rect[1, 1]],
            [rect[2, 0], rect[2, 1]],
            [rect[3, 0], rect[3, 1]],
        ],
        dtype="float32",
    )

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    transform = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, transform, (max_width, max_height))

    # return the warped image
    return warped


def line_intersection(x: cv.typing.MatLike, y: cv.typing.MatLike):
    """
    Calculates the intersection point of two lines, defined by the points
    (x1, y1) and (x2, y2) (first line), and
    (x3, y3) and (x4, y4) (second line).
    If the lines are parallel, (nan, nan) is returned.
    """
    slope_0 = (x[0] - x[1]) * (y[2] - y[3])
    slope_2 = (y[0] - y[1]) * (x[2] - x[3])
    if slope_0 == slope_2:
        # parallel lines
        xis = np.nan
        yis = np.nan
    else:
        xy_01 = x[0] * y[1] - y[0] * x[1]
        xy_23 = x[2] * y[3] - y[2] * x[3]
        denom = slope_0 - slope_2

        xis = (xy_01 * (x[2] - x[3]) - (x[0] - x[1]) * xy_23) / denom
        yis = (xy_01 * (y[2] - y[3]) - (y[0] - y[1]) * xy_23) / denom

    return (xis, yis)


def simplify_polygon(
    in_poly: Polygon,
    length_cutoff: float = 0.15,
    maxiter: int | None = None,
    segment_to_remove: Any | None = None,
):
    """
    Removes segments from a (convex) polygon by continuing neighboring
    segments to a new point of intersection. Purpose is to approximate
    rounded polygons (quadrilaterals) with more sharp-cornered ones.
    """

    x_in = np.asarray(in_poly.exterior.coords)[:-1, 0]
    y_in = np.asarray(in_poly.exterior.coords)[:-1, 1]
    len_poly = len(x_in)
    niter = 0
    if segment_to_remove is not None:
        maxiter = 1
    while len_poly > 4:
        d_in = np.sqrt(
            np.ediff1d(x_in, to_end=x_in[0] - x_in[-1]) ** 2.0
            + np.ediff1d(y_in, to_end=y_in[0] - y_in[-1]) ** 2.0
        )
        d_tot = np.sum(d_in)
        if segment_to_remove is not None:
            k = segment_to_remove
        else:
            k = np.argmin(d_in)
        if d_in[k] < length_cutoff * d_tot:
            ind = generate_point_indices(k - 1, k + 1, len_poly)
            (xis, yis) = line_intersection(x_in[ind], y_in[ind])
            x_in[k] = xis
            y_in[k] = yis
            x_in = np.delete(x_in, (k + 1) % len_poly)
            y_in = np.delete(y_in, (k + 1) % len_poly)
            len_poly = len(x_in)
            niter += 1
            if (maxiter is not None) and (niter >= maxiter):
                break
        else:
            break

    out_poly = Polygon([[ix, iy] for (ix, iy) in zip(x_in, y_in)])

    return out_poly


def generate_point_indices(index_1: Any, index_2: Any, max_len: Any):
    """
    Returns the four indices that give the end points of
    polygon segments corresponding to index_1 and index_2,
    modulo the number of points (max_len).
    """
    return np.array(
        [index_1 % max_len, (index_1 + 1) % max_len, index_2 % max_len, (index_2 + 1) % max_len]
    )


def generate_quad_corners(indices: Any, x: Any, y: Any):
    """
    Returns the four intersection points from the
    segments defined by the x coordinates (x),
    y coordinates (y), and the indices.
    """
    (i, j, k, l) = indices

    def gpi(index_1: Any, index_2: Any):
        return generate_point_indices(index_1, index_2, len(x))

    xis = np.empty(4)
    yis = np.empty(4)
    xis.fill(np.nan)
    yis.fill(np.nan)

    if j <= i or k <= j or l <= k:
        pass
    else:
        (xis[0], yis[0]) = line_intersection(x[gpi(i, j)], y[gpi(i, j)])
        (xis[1], yis[1]) = line_intersection(x[gpi(j, k)], y[gpi(j, k)])
        (xis[2], yis[2]) = line_intersection(x[gpi(k, l)], y[gpi(k, l)])
        (xis[3], yis[3]) = line_intersection(x[gpi(l, i)], y[gpi(l, i)])

    return (xis, yis)


def generate_quad_candidates(in_poly: Polygon):
    """
    Generates a list of bounding quadrilaterals for a polygon,
    using all possible combinations of four intersection points
    derived from four extended polygon segments.
    The number of combinations increases rapidly with the order
    of the polygon, so simplification should be applied first to
    remove very short segments from the polygon.
    """
    # make sure that the points are ordered
    (x_s, y_s) = order_polygon_points(
        np.asarray(in_poly.exterior.coords)[:-1, 0], np.asarray(in_poly.exterior.coords)[:-1, 1]
    )
    x_s_ave = np.average(x_s)
    y_s_ave = np.average(y_s)
    x_shrunk = x_s_ave + 0.9999 * (x_s - x_s_ave)
    y_shrunk = y_s_ave + 0.9999 * (y_s - y_s_ave)
    shrunk_poly = Polygon([[x, y] for (x, y) in zip(x_shrunk, y_shrunk)])
    quads = []
    len_poly = len(x_s)

    for indices in product(range(len_poly), repeat=4):
        (xis, yis) = generate_quad_corners(indices, x_s, y_s)
        if (np.sum(np.isnan(xis)) + np.sum(np.isnan(yis))) > 0:
            # no intersection point for some of the lines
            pass
        else:
            (xis, yis) = order_polygon_points(xis, yis)
            enclose = True
            quad = Polygon([(xis[0], yis[0]), (xis[1], yis[1]), (xis[2], yis[2]), (xis[3], yis[3])])
            if not quad.contains(shrunk_poly):
                enclose = False
            if enclose:
                quads.append(quad)
    return quads


def get_bounding_quad(hull_poly: Polygon):
    """
    Returns the minimum area quadrilateral that contains (bounds)
    the convex hull (openCV format) given as input.
    """
    simple_poly = simplify_polygon(hull_poly)
    bounding_quads = generate_quad_candidates(simple_poly)
    bquad_areas = np.zeros(len(bounding_quads))
    for iquad, bquad in enumerate(bounding_quads):
        bquad_areas[iquad] = bquad.area
    return bounding_quads[np.argmin(bquad_areas)]


def quad_corner_diff(hull_poly: Polygon, bquad_poly: Polygon, region_size: float = 0.9):
    """
    Returns the difference between areas in the corners of a rounded
    corner and the aproximating sharp corner quadrilateral.
    region_size (param) determines the region around the corner where
    the comparison is done.
    """
    bquad_corners = np.zeros((4, 2))
    bquad_corners[:, 0] = np.asarray(bquad_poly.exterior.coords)[:-1, 0]
    bquad_corners[:, 1] = np.asarray(bquad_poly.exterior.coords)[:-1, 1]

    # The point inside the quadrilateral, region_size towards the quad center
    interior_points = np.zeros((4, 2))
    interior_points[:, 0] = np.average(bquad_corners[:, 0]) + region_size * (
        bquad_corners[:, 0] - np.average(bquad_corners[:, 0])
    )
    interior_points[:, 1] = np.average(bquad_corners[:, 1]) + region_size * (
        bquad_corners[:, 1] - np.average(bquad_corners[:, 1])
    )

    # The points p0 and p1 (at each corner) define the line whose intersections
    # with the quad together with the corner point define the triangular
    # area where the roundness of the convex hull in relation to the bounding
    # quadrilateral is evaluated.
    # The line (out of p0 and p1) is constructed such that it goes through the
    # "interior_point" and is orthogonal to the line going from the corner to
    # the center of the quad.
    p0_x = interior_points[:, 0] + (bquad_corners[:, 1] - np.average(bquad_corners[:, 1]))
    p1_x = interior_points[:, 0] - (bquad_corners[:, 1] - np.average(bquad_corners[:, 1]))
    p0_y = interior_points[:, 1] - (bquad_corners[:, 0] - np.average(bquad_corners[:, 0]))
    p1_y = interior_points[:, 1] + (bquad_corners[:, 0] - np.average(bquad_corners[:, 0]))

    corner_area_polys = []
    for i in range(len(interior_points[:, 0])):
        bline = LineString([(p0_x[i], p0_y[i]), (p1_x[i], p1_y[i])])
        corner_area_polys.append(
            Polygon(
                [
                    bquad_poly.intersection(bline).coords[0],
                    bquad_poly.intersection(bline).coords[1],
                    (bquad_corners[i, 0], bquad_corners[i, 1]),
                ]
            )
        )

    hull_corner_area = 0
    quad_corner_area = 0
    for capoly in corner_area_polys:
        quad_corner_area += capoly.area
        hull_corner_area += capoly.intersection(hull_poly).area

    return 1.0 - hull_corner_area / quad_corner_area


def convex_hull_polygon(contour: cv.typing.MatLike):
    """Contour should be a Polygon object."""
    hull = cv.convexHull(contour)
    phull = Polygon([[x, y] for (x, y) in zip(hull[:, :, 0], hull[:, :, 1])])
    return phull


def polygon_form_factor(poly: Polygon):
    # minimum side length
    d_0 = np.amin(np.sqrt(np.sum(np.diff(np.asarray(poly.exterior.coords), axis=0) ** 2.0, axis=1)))
    return poly.area / (poly.length * d_0)


def characterize_card_contour(card_contour: Any, max_segment_area: Any, image_area: Any):
    phull = convex_hull_polygon(card_contour)
    if phull.area < 0.1 * max_segment_area or phull.area < image_area / 1000.0:
        # break after card size range has been explored
        continue_segmentation = False
        is_card_candidate = False
        bounding_poly = None
        crop_factor = 1.0
    else:
        continue_segmentation = True
        bounding_poly = get_bounding_quad(phull)
        qc_diff = quad_corner_diff(phull, bounding_poly)
        crop_factor = min(1.0, (1.0 - qc_diff * 22.0 / 100.0))
        is_card_candidate = bool(
            0.1 * max_segment_area < bounding_poly.area < image_area * 0.99
            and qc_diff < 0.35
            and 0.25 < polygon_form_factor(bounding_poly) < 0.33
        )

    return (continue_segmentation, is_card_candidate, bounding_poly, crop_factor)


def contour_image_gray(full_image: cv.typing.MatLike, thresholding: str = "adaptive"):
    gray = cv.cvtColor(full_image, cv.COLOR_BGR2GRAY)
    if thresholding == "adaptive":
        fltr_size = 1 + 2 * (min(full_image.shape[0], full_image.shape[1]) // 20)
        thresh = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, fltr_size, 10
        )
    else:
        _, thresh = cv.threshold(gray, 70, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(
        cast(cv.typing.MatLike, np.uint8(thresh)),
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE,
    )
    return contours


def contour_image_rgb(full_image: cv.typing.MatLike) -> Sequence[cv.typing.MatLike]:
    blue, green, red = cv.split(full_image)
    blue = CLAHE.apply(blue)
    green = CLAHE.apply(green)
    red = CLAHE.apply(red)
    _, thr_b = cv.threshold(blue, 110, 255, cv.THRESH_BINARY)
    _, thr_g = cv.threshold(green, 110, 255, cv.THRESH_BINARY)
    _, thr_r = cv.threshold(red, 110, 255, cv.THRESH_BINARY)
    contours_b, _ = cv.findContours(
        cast(cv.typing.MatLike, np.uint8(thr_b)),
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE,
    )
    contours_g, _ = cv.findContours(
        cast(cv.typing.MatLike, np.uint8(thr_g)),
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE,
    )
    contours_r, _ = cv.findContours(
        cast(cv.typing.MatLike, np.uint8(thr_r)),
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE,
    )
    return list(contours_b) + list(contours_g) + list(contours_r)


def contour_image(full_image: cv.typing.MatLike, mode: str = "gray"):
    if mode == "gray":
        contours = contour_image_gray(full_image, thresholding="simple")
    elif mode == "adaptive":
        contours = contour_image_gray(full_image, thresholding="adaptive")
    elif mode == "rgb":
        contours = contour_image_rgb(full_image)
    elif mode == "all":
        contours = list(contour_image_gray(full_image, thresholding="simple"))
        contours += contour_image_gray(full_image, thresholding="adaptive")
        contours += contour_image_rgb(full_image)
    else:
        raise ValueError("Unknown segmentation mode.")
    contours_sorted = sorted(contours, key=cv.contourArea, reverse=True)
    return contours_sorted


def phash_diff(phashes: list[ImageHash], phash_im: ImageHash):
    """
    Calculates the phash difference between the given phash and
    each of the reference images.
    """
    diff = np.zeros(len(phashes))
    for i, ref_phash in enumerate(phashes):
        diff[i] = phash_im - ref_phash
    return diff


def rotate_image(image: cv.typing.MatLike, angle: float) -> cv.typing.MatLike:
    # from scipy.ndimage import rotate

    # rotated_image = rotate(image, angle)
    # return rotated_image

    w, h = image.shape[1::-1]
    center = tuple(np.array((w, h)) / 2)
    rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    rotated_image = cv.warpAffine(image, rot_mat, (new_w, new_h), flags=cv.INTER_LINEAR)
    rotated_image = rotated_image[1:-1, 1:-1]
    return rotated_image


def phash_compare(data: dict[str, ImageHash], im_seg: cv.Mat):
    """
    Runs perceptive hash comparison between given image and
    the (pre-hashed) reference set.
    """
    is_recognized = False
    rotations = [0.0, 90.0, 180.0, 270.0]
    n_rotations = len(rotations)
    hash_separation_thr = 4.0
    best_score = 0.0
    best_gid = "unknown"
    n_cards = len(data)
    gids = list(data.keys())
    phashes = list(data.values())

    d_0_dist = np.zeros(n_rotations)
    d_0 = np.zeros((n_cards, n_rotations))
    for j, rot in enumerate(rotations):
        if not -1.0e-5 < rot < 1.0e-5:
            rotated = rotate_image(im_seg, rot)
            arr = np.uint8(255 * cv.cvtColor(rotated, cv.COLOR_BGR2RGB))
            phash_im = phash(Image.fromarray(arr), hash_size=32)
        else:
            arr = np.uint8(255 * cv.cvtColor(im_seg, cv.COLOR_BGR2RGB))
            phash_im = phash(Image.fromarray(arr), hash_size=32)
        d_0[:, j] = phash_diff(phashes, phash_im)
        d_0_ = d_0[d_0[:, j] > np.amin(d_0[:, j]), j]
        d_0_ave = np.average(d_0_)
        d_0_std = np.std(d_0_)
        d_0_dist[j] = (d_0_ave - np.amin(d_0[:, j])) / d_0_std
        if d_0_dist[j] > hash_separation_thr and np.argmax(d_0_dist) == j:
            is_recognized = True
            recognition_score = d_0_dist[j] / hash_separation_thr
            if recognition_score > best_score:
                best_score = recognition_score
                best_gid = gids[np.argmin(d_0[:, j])]
            continue
    return (is_recognized, best_score, best_gid)


@dataclass
class CardCandidate:
    image: cv.typing.MatLike
    bounding_quad: Polygon
    image_area_fraction: float
    is_recognized: bool = False
    recognition_score: float = 0.0
    is_fragment: bool = False
    gid: str = "unknown"
    uri: str = "unknown"

    def contains(self, other: CardCandidate):
        return bool(other.bounding_quad.within(self.bounding_quad) and other.gid == self.gid)


def best_matches(
    db: dict[str, Any],
    data: dict[str, ImageHash],
    filename: str | Path,
) -> list[CardCandidate]:
    img = cv.imread(str(filename))
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    lightness, redness, yellowness = cv.split(lab)
    corrected_lightness = CLAHE.apply(lightness)
    limg = cv.merge((corrected_lightness, redness, yellowness))
    adjust = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    # alg_list = ["adaptive", "rgb", "gray"]
    alg = "all"

    candidate_list = []

    # for alg in alg_list:
    full_image = adjust.copy()
    image_area = full_image.shape[0] * full_image.shape[1]
    max_segment_area = 0.01  # largest card area
    contours = contour_image(full_image, mode=alg)
    for card_contour in contours:
        try:
            (
                continue_segmentation,
                is_card_candidate,
                bounding_poly,
                crop_factor,
            ) = characterize_card_contour(card_contour, max_segment_area, image_area)
        except NotImplementedError:
            # this can occur in Shapely for some funny contour shapes
            # resolve by discarding the candidate
            (continue_segmentation, is_card_candidate, bounding_poly, crop_factor) = (
                True,
                False,
                None,
                1.0,
            )
        if not continue_segmentation:
            break
        if is_card_candidate:
            assert bounding_poly is not None
            if max_segment_area < 0.1:
                max_segment_area = bounding_poly.area
            scaled = scale(
                bounding_poly,
                xfact=crop_factor,
                yfact=crop_factor,
                origin="centroid",
            )
            warped = four_point_transform(full_image, scaled)
            candidate = CardCandidate(warped, bounding_poly, bounding_poly.area / image_area)
            candidate_list.append(candidate)

    for candidate in candidate_list:
        im_seg = candidate.image

        # Easy fragment / duplicate detection
        for other_candidate in candidate_list:
            if other_candidate.is_recognized and not other_candidate.is_fragment:
                if other_candidate.contains(candidate):
                    candidate.is_fragment = True
        if not candidate.is_fragment:
            (
                candidate.is_recognized,
                candidate.recognition_score,
                candidate.gid,
            ) = phash_compare(data, im_seg)

    recognized_list = []
    unique_gids = set()
    for candidate in candidate_list:
        if candidate.is_recognized and not candidate.is_fragment:
            if candidate.gid not in unique_gids:
                gid = candidate.gid
                if any(gid.endswith(face) for face in ("back", "front")):
                    gid, _ = gid.rsplit("-", 1)
                candidate.uri = cast(dict[str, Any], db)[gid]["uri"]
                recognized_list.append(candidate)
                unique_gids.add(gid)

    ranked = sorted(recognized_list, key=lambda x: x.recognition_score, reverse=True)
    return list(ranked)


@dataclass
class Result:
    name: str
    target: str
    uri: str
    score: float


def match_results(
    db: dict[str, Any],
    data: dict[str, ImageHash],
    file: str | Path,
) -> list[Result]:
    assert data
    assert db
    matches = best_matches(db, data, file)
    if not matches:
        return []

    results: list[Result] = []
    for match in matches:
        gid = match.gid
        if any(match.gid.endswith(face) for face in ("back", "front")):
            gid, face = match.gid.rsplit("-", 1)
            target = Path(IMAGES_DIR) / f"{gid}-{face}.jpg"
        else:
            target = Path(IMAGES_DIR) / f"{match.gid}.jpg"
        name = cast(dict[str, Any], db)[gid]["name"]
        uri = cast(dict[str, Any], db)[gid]["scryfall_uri"]
        results.append(
            Result(
                name=name,
                target=str(target),
                uri=uri,
                score=match.recognition_score,
            )
        )

    return results
