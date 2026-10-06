"""
zone_connectivity.py
========================
Geometric zone connectivity normalization.

Invariant: every zone ID occupies exactly one 6-connected voxel component;
every in-hull voxel belongs to exactly one zone; voxel count conserved.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi

from bulkhead_placement import (
    BulkheadLayout,
    Zone,
    ZoneType,
    EPS_EMPTY,
)

STRUCTURE_6 = ndi.generate_binary_structure(3, 1)
_FACE_OFFSETS = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)
_SIDE_TYPES = frozenset({
    ZoneType.SIDE_DB, ZoneType.SIDE_LOWER, ZoneType.SIDE_UPPER,
})


def count_zone_components(
    zone_mask: np.ndarray,
    hull_occ: np.ndarray,
    zone_id: int,
) -> Tuple[int, List[np.ndarray]]:
    """Return (n_components, list of boolean masks per component)."""
    binary = (zone_mask == zone_id) & hull_occ
    if not binary.any():
        return 0, []
    labelled, n = ndi.label(binary, structure=STRUCTURE_6)
    comps = [(labelled == cc) for cc in range(1, n + 1)]
    comps.sort(key=lambda m: -int(m.sum()))
    return n, comps


def structural_compatible(template: Zone, recipient: Zone) -> bool:
    """Face-adjacent recipient must share structural role."""
    if template.zone_type != recipient.zone_type:
        return False
    if template.side != "centre" and recipient.side != template.side:
        return False
    if template.tier_role != recipient.tier_role:
        return False
    if template.zone_type in _SIDE_TYPES and template.hold_idx >= 0:
        if recipient.hold_idx >= 0 and template.hold_idx != recipient.hold_idx:
            return False
    return True


def _hull_occ(zone_mask: np.ndarray) -> np.ndarray:
    return zone_mask >= 0


def _zone_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, 0, 0, 0, 0, 0
    return (
        int(coords[:, 0].min()), int(coords[:, 0].max()) + 1,
        int(coords[:, 1].min()), int(coords[:, 1].max()) + 1,
        int(coords[:, 2].min()), int(coords[:, 2].max()) + 1,
    )


def _rebuild_zone_from_mask(
    zone_id: int,
    voxel_mask: np.ndarray,
    template: Zone,
    fm: np.ndarray,
) -> Zone:
    """Build a Zone from an exact voxel subset, copying template metadata."""
    x0, x1, y0, y1, z0, z1 = _zone_bbox_from_mask(voxel_mask)
    region = fm[x0:x1, y0:y1, z0:z1]
    local = voxel_mask[x0:x1, y0:y1, z0:z1]
    inside = local & (region > EPS_EMPTY)
    avail = int(inside.sum())
    total = (x1 - x0) * (y1 - y0) * (z1 - z0)
    nx, ny, nz_total = fm.shape

    if avail > 0:
        occ = np.argwhere(inside)
        cx = (x0 + float(occ[:, 0].mean()) + 0.5) / nx
        cy = (y0 + float(occ[:, 1].mean()) + 0.5) / ny
        cz = (z0 + float(occ[:, 2].mean()) + 0.5) / nz_total
        eff = float(region[inside].sum())
    else:
        cx = ((x0 + x1) / 2) / nx
        cy = ((y0 + y1) / 2) / ny
        cz = ((z0 + z1) / 2) / nz_total
        eff = 0.0

    hull_avail = eff / max(1, total)
    return Zone(
        zone_id=zone_id,
        zone_type=template.zone_type,
        hold_idx=template.hold_idx,
        x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
        available_cells=avail,
        total_cells=total,
        eligible_comps=template.eligible_comps,
        cx_norm=round(cx, 5),
        cy_norm=round(cy, 5),
        cz_norm=round(cz, 5),
        hull_avail=round(hull_avail, 5),
        zone_width_norm=round((y1 - y0) / ny, 5),
        zone_length_norm=round((x1 - x0) / nx, 5),
        deck_idx=template.deck_idx,
        tier_role=template.tier_role,
        side=template.side,
        mirror_id=template.mirror_id,
    )


def _best_face_neighbor_zone(
    ix: int, iy: int, iz: int,
    zone_mask: np.ndarray,
    self_zid: int,
    template: Zone,
    templates: Dict[int, Zone],
) -> Optional[int]:
    nx, ny, nz = zone_mask.shape
    tallies: Counter = Counter()
    for dx, dy, dz in _FACE_OFFSETS:
        jx, jy, jz = ix + dx, iy + dy, iz + dz
        if not (0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz):
            continue
        nzid = int(zone_mask[jx, jy, jz])
        if nzid < 0 or nzid == self_zid:
            continue
        recip = templates.get(nzid)
        if recip is None:
            continue
        if not structural_compatible(template, recip):
            continue
        tallies[nzid] += 1
    if not tallies:
        return None
    best, count = tallies.most_common(1)[0]
    if len(tallies) > 1 and tallies.most_common(2)[1][1] == count:
        return None
    return best


def normalize_zone_connectivity(layout: BulkheadLayout) -> BulkheadLayout:
    """
    Split or reassign zone IDs so each occupies one 6-connected component.
    Voxel count conserved; zone_mask fully rebuilt.
    """
    fm = layout.hull_result.full_mask
    zm = layout.zone_mask.copy()
    hull_occ = _hull_occ(zm)
    voxels_before = int(hull_occ.sum())

    templates: Dict[int, Zone] = {z.zone_id: z for z in layout.zones}
    next_id = max(templates.keys(), default=-1) + 1

    for zid in sorted({int(z) for z in np.unique(zm) if z >= 0}):
        if zid not in templates:
            continue
        template = templates[zid]
        n_cc, comps = count_zone_components(zm, hull_occ, zid)
        if n_cc <= 1:
            continue

        for comp in comps[1:]:
            nvox = int(comp.sum())
            if nvox == 1:
                coords = np.argwhere(comp)
                ix, iy, iz = int(coords[0, 0]), int(coords[0, 1]), int(coords[0, 2])
                recipient = _best_face_neighbor_zone(
                    ix, iy, iz, zm, zid, template, templates,
                )
                if recipient is not None:
                    zm[ix, iy, iz] = recipient
                    continue
            new_zid = next_id
            next_id += 1
            zm[comp] = new_zid
            templates[new_zid] = replace(template, zone_id=new_zid, mirror_id=-1)

    voxels_after = int(_hull_occ(zm).sum())
    if voxels_before != voxels_after:
        raise AssertionError(f"voxel loss: {voxels_before} -> {voxels_after}")

    new_zones: List[Zone] = []
    for zid in sorted({int(z) for z in np.unique(zm) if z >= 0}):
        mask = (zm == zid)
        tmpl = templates.get(zid)
        if tmpl is None:
            continue
        z = _rebuild_zone_from_mask(zid, mask, tmpl, fm)
        if z.available_cells == 0:
            raise AssertionError(f"zone {zid} has zero cells after normalization")
        new_zones.append(z)

    out = BulkheadLayout(
        zones=new_zones,
        zone_mask=zm,
        transverse_bulkheads_x=layout.transverse_bulkheads_x,
        n_holds=layout.n_holds,
        params=layout.params,
        hull_result=layout.hull_result,
        side_carve_mode=layout.side_carve_mode,
    )
    assert_zone_connectivity_invariants(out)
    return out


def assert_zone_connectivity_invariants(layout: BulkheadLayout) -> None:
    """Hard checks after normalization."""
    zm = layout.zone_mask
    hull_occ = _hull_occ(zm)
    fm = layout.hull_result.full_mask

    occupied = fm > EPS_EMPTY
    if not np.array_equal(hull_occ, occupied):
        gap = int((occupied & ~hull_occ).sum())
        extra = int((hull_occ & ~occupied).sum())
        raise AssertionError(
            f"hull/zoning mismatch: unzoned occupied={gap} zoned empty={extra}"
        )

    vox_total = int(hull_occ.sum())
    zone_vox = sum(z.available_cells for z in layout.zones)
    if zone_vox != vox_total:
        raise AssertionError(
            f"volume mismatch: zones sum={zone_vox} hull={vox_total}"
        )

    for z in layout.zones:
        n_cc, _ = count_zone_components(zm, hull_occ, z.zone_id)
        if n_cc != 1:
            raise AssertionError(
                f"zone {z.zone_id} ({z.zone_type.name}) has {n_cc} components"
            )
