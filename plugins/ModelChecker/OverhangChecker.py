# Copyright (c) 2026 UltiMaker
# Cura is released under the terms of the LGPLv3 or higher.

import math
from typing import Set, Tuple

import numpy

from UM.Application import Application
from UM.Scene.SceneNode import SceneNode
from cura.Settings.GlobalStack import GlobalStack


MIN_SUPPORT_AREA: float = 0.1  # mm^2   # TODO: Change via preference?
CLOSE_TO_BUILDPLATE: float = 0.05  # TODO: This needs to be proportional to (half) the layer-height.


def getSupportAngle() -> float:
    # NOTE/TODO: This duplicates some code from PaintView, which means we got this code in some form 3x.
    global_container_stack: GlobalStack = Application.getInstance().getGlobalContainerStack()
    if not global_container_stack or not global_container_stack.getValue("support_enable"):
        return 1.0
    extruder_nr = int(global_container_stack.getExtruderPositionValueWithDefault("support_extruder_nr"))
    if extruder_nr < 0 or extruder_nr >= len(global_container_stack.extruderList):
        return 1.0
    return (global_container_stack.extruderList[extruder_nr].getProperty("support_angle", "value") or 90.0) * math.tau / 360.0


def checkForDownFaces(node: SceneNode, support_angle: float) -> bool:
    meshdata = node.getMeshDataTransformed()
    face_count = meshdata.getFaceCount()
    candidate_overhangs = {}
    for i_face in range(face_count):
        pos, face_norm = meshdata.getFacePlane(i_face)

        # Check the angle.
        angle = -math.asin(face_norm[1]) if -1.0 <= face_norm[1] <= 1.0 else 0.0
        if angle < support_angle:
            continue

        # Check if the face is too close to (or underneath) the build-plate to 'count'.
        a, b, c = meshdata.getFaceNodes(i_face)
        if max(float(a[1]), float(b[1]), float(c[1])) < CLOSE_TO_BUILDPLATE:
            continue

        # Collect the area for further analysis.
        area = 0.5 * numpy.linalg.norm(numpy.abs(numpy.cross(b - a, c - a)))
        candidate_overhangs[i_face] = area

    if len(candidate_overhangs) <= 0:
        return False

    visited = set()
    def _collect_neighbours(face_idx: int) -> Set:
        result = {face_idx}
        visited.add(face_idx)
        nb_face_ids = meshdata.getFaceNeighbourIDs(face_idx)
        for nb_face in nb_face_ids:
            if nb_face not in candidate_overhangs or nb_face in visited:
                continue
            result = result.union(_collect_neighbours(nb_face))
        return result

    # Is there a big enough area that needs to be supported?
    for i_face, angle in candidate_overhangs.items():
        if i_face in visited:
            continue
        group = _collect_neighbours(i_face)
        group_area = sum([candidate_overhangs[x] for x in group])
        if group_area > MIN_SUPPORT_AREA:
            return True

    return False


def checkForDownVertices(node: SceneNode) -> bool:
    meshdata = node.getMeshDataTransformed()
    face_count = meshdata.getFaceCount()

    def _to_hashable(pt: numpy.ndarray) -> Tuple[float, float, float]:
        return float(pt[0]), float(pt[1]), float(pt[2])

    verts_with_lower = dict()
    def _handle_edge(va: numpy.ndarray, vb: numpy.ndarray) -> None:
        if min(float(va[1]), float(vb[1])) < CLOSE_TO_BUILDPLATE:
            verts_with_lower[_to_hashable(va)] = False
            verts_with_lower[_to_hashable(vb)] = False
        if va[1] == vb[1]:
            return
        verts_with_lower[_to_hashable(va if va[1] > vb[1] else vb)] = False

    # Create a vertex adjacency graph -- but only append vertices that are _lower_ (except too close or below the BP).
    for i_face in range(face_count):
        a, b, c = meshdata.getFaceNodes(i_face)
        for vert in (a, b, c):
            v = _to_hashable(vert)
            if v not in verts_with_lower:
                verts_with_lower[v] = True
        _handle_edge(a, b)
        _handle_edge(b, c)
        _handle_edge(c, a)

    # Any vertex that has no adjacencies (that is, no vertices that are lower than it) is downward.
    return any(verts_with_lower.values())
