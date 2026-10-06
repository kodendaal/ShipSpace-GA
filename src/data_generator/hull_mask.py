"""
hull_mask.py
============
Stage 2: Hull Mask Generation
------------------------------
Produces hull_mask[x, y, z] ∈ [0.0, 1.0] for each ship instance.

  1.0  = voxel fully inside hull envelope
  0.0  = voxel fully outside hull
  0.0-1.0 = boundary voxel (partially inside)

Two generation paths:
  A) ParametricHullMask  — analytical, from ShipParameterization geometry
  B) STLHullMask         — mesh-based, from .stl file via trimesh voxelisation;
     the generator uses CachedSTLHullMask (stl_mask_cache), which reads the
     stored voxelisations in data/hull_cache instead of the meshes

Both return the same HullMaskResult dataclass so all downstream
stages (bulkhead placement, compartment assignment, graph construction)
are agnostic to hull source.

Superstructure mask is generated separately and appended in z above
nz_hull layers.

Assumptions:
  - Hull is symmetric about the transverse centreline (y = ny/2 * voxel_size)
  - STL meshes are watertight (closed manifold), oriented upright:
      x = longitudinal (aft=min, bow=max in mesh space — normalised during import)
      y = transverse
      z = vertical (keel=0)
  - STL meshes cover main hull only (keel to main deck), no superstructure
  - Cell sizes dx, dy, dz come from ShipParameterization

Dependencies: numpy (trimesh only to voxelise STL meshes)
"""

from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
try:
    import trimesh
except ImportError:
    trimesh = None  # STL path unavailable without trimesh

from ship_params import ShipParameterization, ShipType, ss_logical_deck_index


# Minimum fractional occupancy at which a voxel is considered part of the hull
# for geometric decisions (e.g. top deck detection, zone assignment).
# Voxels below this are treated as empty; voxels above remain in the hull,
# even if they are only partially filled.
EPS_EMPTY = 0.10


# Hull library. Each hull form is an STL mesh whose fractional voxelisation is
# stored in data/hull_cache; the cached grids alone are enough to generate.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HULL_CACHE_DIR = DATA_DIR / "hull_cache"
STL_LIBRARY_DIR = DATA_DIR / "stl_library"


def _hull_forms(family: str) -> list:
    """Hull forms of one family, sorted by name; the paths name the STL meshes."""
    names = {path.stem for path in STL_LIBRARY_DIR.glob(f"{family}_*.stl")}
    names.update(path.name.rsplit("_", 2)[0]
                 for path in HULL_CACHE_DIR.glob(f"{family}_*.npy"))
    return [STL_LIBRARY_DIR / f"{name}.stl" for name in sorted(names)]


transport_stls = _hull_forms("transport")
work_stls = _hull_forms("work")
defense_stls = _hull_forms("defense")
yacht_stls = _hull_forms("yacht")

# Map each ShipType to the list of candidate hull forms
SHIPTYPE_TO_STL_LIB = {
    ShipType.BULKER:   transport_stls,
    ShipType.TANKER:   transport_stls,
    ShipType.CARGO:    transport_stls,
    ShipType.OSV:      work_stls,
    ShipType.YACHT:    yacht_stls,
    ShipType.PATROL:   defense_stls,
}

# ─────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────

@dataclass
class HullMaskResult:
    """
    Complete voxel availability mask for one ship instance.

    full_mask[nx, ny, nz_total] is the single source of truth — 1.0 where a cell is available, 0.0 outside.

    The SS is placed AFTER the hull mask is computed by stacking nz_ss
    layers above top_deck_z for each column in the SS footprint.
    Like Tetris — the SS settles onto whatever surface the hull provides,
    whether a forecastle, tapered bow, or flat main deck.

      full_mask[:, :, :nz_hull]  = hull cells
      full_mask[:, :, nz_hull:]  = superstructure cells
      top_deck_z[nx, ny]         = index of highest occupied hull cell
                                   per column (-1 if column is empty)

    hull_source : 'parametric' | 'stl:<filename>'
    """

    full_mask:   np.ndarray          # (nx, ny, nz_total)
    top_deck_z:  np.ndarray          # (nx, ny) int
    params:      ShipParameterization
    hull_source: str = "parametric"
    ss_only:     Optional[np.ndarray] = None  # same shape as full_mask, 1 where SS, 0 else
    hull_fraction: Optional[np.ndarray] = None  # (nx, ny, nz_hull) occupancy before SS placement

    @property
    def nz_hull(self) -> int:
        return self.params.nz_hull

    @property
    def hull_mask(self) -> np.ndarray:
        """
        Hull cells only (excludes superstructure).
        """
        if self.ss_only is None:
            # no separate SS mask: assume SS is only z >= nz_hull
            return self.full_mask[:, :, :self.nz_hull]
        hull = self.full_mask - self.ss_only
        # numerical safety
        return np.clip(hull, 0.0, 1.0)

    @property
    def ss_mask(self) -> np.ndarray:
        """
        Superstructure cells only (where ss_only was written).
        """
        if self.ss_only is None:
            # no separate SS mask
            return self.full_mask[:, :, self.nz_hull:]
        return self.ss_only

    @property
    def hull_volume_voxels(self) -> float:
        return float(self.hull_mask.sum())

    @property
    def ss_volume_voxels(self) -> float:
        return float(self.ss_mask.sum())

    @property
    def hull_volume_m3(self) -> float:
        return self.hull_volume_voxels * self.params.cell_volume

    def midplane_slice(self) -> np.ndarray:
        """Side view at centreline: full_mask[:, ny//2, :] ->(nx, nz_total)."""
        return self.full_mask[:, self.params.ny // 2, :]

    def topdown_slice(self, z_idx: int = 1) -> np.ndarray:
        """Plan view at deck z_idx: ->(nx, ny)."""
        return self.full_mask[:, :, min(z_idx, self.full_mask.shape[2] - 1)]

    def summary(self) -> str:
        p = self.params
        nz_t = self.full_mask.shape[2]
        fill = self.hull_volume_voxels / max(1, p.nx * p.ny * p.nz_hull)
        valid = self.top_deck_z[self.top_deck_z >= 0]
        fc_delta = int(valid.max() - valid.min()) if len(valid) else 0
        lines = [
            f"HullMask  source={self.hull_source}",
            f"  grid      : {p.nx} x {p.ny} x {nz_t}  (hull {p.nz_hull} + SS {p.nz_ss})",
            f"  cell size : dx={p.dx_m:.3f}  dy={p.dy_m:.3f}  dz={p.dz_m:.3f} m"
            f"  ({'anisotropic' if p.anisotropic else 'isotropic'})",
            f"  hull vol  : {self.hull_volume_m3:.0f} m3  ({self.hull_volume_voxels:.0f} cells)",
            f"  hull fill : {fill*100:.1f}% of bounding box",
            f"  SS cells  : {self.ss_volume_voxels:.0f} vox",
            f"  deck range: z={int(valid.min()) if len(valid) else '?'} to "
            f"z={int(valid.max()) if len(valid) else '?'}  "
            f"(forecastle delta = {fc_delta} voxels)",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Surface computation and superstructure placement
# ─────────────────────────────────────────────────────────────────

def compute_top_deck_z(hull_mask: np.ndarray) -> np.ndarray:
    """
    For each (x, y) column find the z-index of the highest occupied cell.
    Reads directly from hull geometry — captures forecastle, tapered bow,
    any hull form. No parameters needed.

    Returns int array (nx, ny); columns with no cells return -1.
    """
    # Any voxel with fractional occupancy above EPS_EMPTY is considered
    # part of the hull for the purpose of detecting the local deck height.
    occupied = hull_mask > EPS_EMPTY          # (nx, ny, nz)
    nz = hull_mask.shape[2]
    # argmax on reversed z gives first True = last True in original
    top = np.argmax(occupied[:, :, ::-1], axis=2)
    has_any = occupied.any(axis=2)
    top_z = (nz - 1 - top).astype(np.int32)
    top_z[~has_any] = -1
    return top_z


def place_superstructure(hull_mask: np.ndarray,
                          top_deck_z: np.ndarray,
                          p: ShipParameterization) -> np.ndarray:
    """
    Stack nz_ss z-layers above the hull surface — true Tetris.

    Each column's superstructure settles directly onto the local hull deck:
    layer ``level`` sits at ``top_deck_z[x, y] + 1 + level``.  The SS therefore
    follows the real deck (forecastle, sheer, deck steps) exactly — no fake
    hull fill.  Logical decks (for taper and the compartment stack) are indexed
    via ``ss_logical_deck_index`` so taper is applied per accommodation deck,
    not per z-layer (correct at fine, anisotropic ``dz``).

    Returns full_mask (nx, ny, nz_total) with hull and SS cells combined.
    """
    nx, ny, nz_hull = hull_mask.shape
    nz_ss    = p.nz_ss
    nz_total = nz_hull + nz_ss

    full = np.zeros((nx, ny, nz_total), dtype=np.float32)
    full[:, :, :nz_hull] = hull_mask

    ss_only = np.zeros((nx, ny, nz_total), dtype=np.float32)

    if nz_ss == 0:
        return full, ss_only

    base_x0 = p.ss_x_start
    base_x1 = p.ss_x_end
    base_y0 = p.ss_y_start
    base_y1 = p.ss_y_end


    base_nx = max(1, base_x1 - base_x0)
    base_ny = max(1, base_y1 - base_y0)

    for level in range(nz_ss):
        logical = ss_logical_deck_index(level, p.ss_layers_per_deck)
        x_shrink = int(round(logical * p.ss_x_taper_per_level * base_nx / 2))
        y_shrink = int(round(logical * p.ss_y_taper_per_level * base_ny / 2))
        x0 = min(base_x0 + x_shrink, base_x0 + base_nx // 2)
        x1 = max(base_x1 - x_shrink, x0 + 1)
        y0 = min(base_y0 + y_shrink, base_y0 + base_ny // 2)
        y1 = max(base_y1 - y_shrink, y0 + 1)

        for xi in range(x0, x1):
            for yi in range(y0, y1):
                if xi >= nx or yi >= ny:
                    continue
                tdz = int(top_deck_z[xi, yi])
                if tdz < 0:
                    continue           # no hull here — skip
                z_ss = tdz + 1 + level
                if z_ss < nz_total:
                    full[xi, yi, z_ss] = 1.0
                    ss_only[xi, yi, z_ss] = 1.0

    return full, ss_only

# Path A: Parametric hull mask
# ─────────────────────────────────────────────────────────────────

class ParametricHullMask:
    """
    Generates a hull mask analytically from ShipParameterization geometry.

    Hull form model
    ---------------
    Three regions along x (aft→bow):

      [0 .. stern_end]       Stern taper zone
      [stern_end .. bow_start] Midship: full rectangular section
      [bow_start .. nx-1]    Bow taper zone

    In each taper zone, the available half-breadth is reduced using a
    cosine blend:
      half_breadth(x) = (B/2) x 0.5 x (1 - cos(π x t))
    where t ∈ [0,1] is the normalised position within the taper zone.

    Vertical (z) profile
    --------------------
    Two contributions:

    1. Underwater taper (deadrise): below design waterline, the hull
       narrows toward the keel. Modelled as a linear taper from full
       half-breadth at waterline to deadrise_frac x B/2 at keel.

    2. Above waterline: full half-breadth (flat topsides).

    This produces a realistic prismatic mid-body with tapered ends and
    modest deadrise — sufficient for synthetic compartment generation.

    Boundary voxels (partially inside hull) are handled by computing
    what fraction of the voxel's y-extent falls within the available
    half-breadth. This gives hull_avail ∈ (0,1) for boundary cells.
    """

    def __init__(self, deadrise_deg: float = 12.0):
        """
        Parameters
        ----------
        deadrise_deg : half-angle of deadrise at keel (degrees).
            Typical values: 5-8° tanker, 10-15° cargo, 15-20° yacht.
            Default 12° is a reasonable midpoint for parametric generation.
        """
        self.deadrise_deg = deadrise_deg

    def build(self, p: ShipParameterization) -> HullMaskResult:
        """
        Compute hull_mask (nx, ny, nz_hull) for parametric hull.
        """
        nx, ny, nz = p.nx, p.ny, p.nz_hull
        vs = p.voxel_size
        dy = p.dy_m   # transverse cell size (== vs in isotropic mode)
        dz = p.dz_m   # vertical cell size   (== vs in isotropic mode)

        # Full half-breadth at midship — in TRANSVERSE (y) voxel units.
        B_half_m = p.B / 2.0           # metres
        B_half_v = B_half_m / dy        # y-voxel units (anisotropic-correct)

        # Design waterline z-index (continuous) — in VERTICAL (z) voxel units
        T_v = p.T / dz                  # draft in z-voxel units

        # ── Longitudinal taper profile ────────────────────────────────
        # half_breadth_x[i] = available half-breadth (voxel units) at x-column i
        half_bx = self._longitudinal_taper(nx, B_half_v, p)

        # ── Vertical taper profile ────────────────────────────────────
        # For each z layer, how much does deadrise reduce the half-breadth?
        # taper_z[k] ∈ [0,1]: multiplier applied to half_bx
        taper_z = self._vertical_taper(nz, T_v, p)

        # ── Voxel centre positions ────────────────────────────────────
        # y_centres[j] = distance from centreline (in voxel units) for column j
        # Centreline is at y = ny/2 voxels
        y_idx = np.arange(ny, dtype=float)
        y_centres = np.abs(y_idx + 0.5 - ny / 2.0)   # distance from CL in vox units

        # ── Hull availability computation ─────────────────────────────
        # For each (x, y, z) voxel:
        #   available_hb = half_bx[x] * taper_z[z]
        #   voxel spans [y_centre-0.5, y_centre+0.5] in voxel units
        #   hull_avail = clamp(available_hb - (y_centre-0.5), 0, 1)
        #
        # This correctly produces:
        #   1.0  for voxels fully inside (y_centre+0.5 ≤ available_hb)
        #   0.0  for voxels fully outside (y_centre-0.5 ≥ available_hb)
        #   frac for boundary voxels

        # Broadcast shapes: (nx,1,nz) x (1,ny,1) ->(nx,ny,nz)
        hbx = half_bx[:, np.newaxis, np.newaxis]         # (nx,1,1)
        tz  = taper_z[np.newaxis, np.newaxis, :]          # (1,1,nz)
        yc  = y_centres[np.newaxis, :, np.newaxis]        # (1,ny,1)

        avail_hb = hbx * tz                               # (nx,1,nz)

        # Fraction of this voxel (in y) that is inside the hull
        inside_frac = np.clip(avail_hb - (yc - 0.5), 0.0, 1.0)

        hull_mask = inside_frac.astype(np.float32)

        # ── Surface + superstructure ──────────────────────────────────
        # Read the hull surface as-is — forecastle is in the hull mesh.
        # For the parametric path nz_hull is the flat midship deck height
        # (forecastle is not modelled parametrically; use STL for that).
        top_deck_z = compute_top_deck_z(hull_mask)
        full_mask, ss_only  = place_superstructure(hull_mask, top_deck_z, p)

        return HullMaskResult(
            full_mask=full_mask,
            top_deck_z=top_deck_z,
            params=p,
            hull_source="parametric",
            ss_only=ss_only,
            hull_fraction=hull_mask,
        )

    # ── Internal helpers ──────────────────────────────────────────────

    def _longitudinal_taper(self, nx: int, B_half_v: float,
                             p: ShipParameterization) -> np.ndarray:
        """
        Returns half_bx[i] for i in range(nx).

        Stern taper: columns 0 .. stern_end (aft ->stern_end)
        Midship:     columns stern_end .. bow_start (full breadth)
        Bow taper:   columns bow_start .. nx-1

        Uses cosine blend: smoother than linear, avoids sharp corners
        that would create unphysical compartment shapes.
        """
        half_bx = np.full(nx, B_half_v, dtype=float)

        # Stern zone: x=0 is aft tip (half-breadth ->0)
        stern_end = max(1, int(round(p.stern_taper_frac * nx)))
        for i in range(stern_end):
            t = i / stern_end                       # 0=aft tip, 1=end of stern taper
            half_bx[i] = B_half_v * 0.5 * (1 - np.cos(np.pi * t))

        # Bow zone: x=nx-1 is bow tip (half-breadth ->0)
        bow_start = max(stern_end + 1, nx - int(round(p.bow_taper_frac * nx)))
        for i in range(bow_start, nx):
            t = (nx - 1 - i) / max(1, (nx - 1 - bow_start))  # 1=bow_start, 0=bow tip
            t = np.clip(t, 0.0, 1.0)
            half_bx[i] = B_half_v * 0.5 * (1 - np.cos(np.pi * t))

        return half_bx

    def _vertical_taper(self, nz: int, T_v: float,
                         p: ShipParameterization) -> np.ndarray:
        """
        Returns taper_z[k] ∈ [0,1] for k in range(nz).

        Below waterline: linear deadrise reduces half-breadth from
        1.0 at waterline to deadrise_frac at keel (z=0).

        Above waterline: flat topsides, taper_z = 1.0.

        deadrise_frac: fraction of full half-breadth remaining at keel.
        Derived from deadrise angle and half-breadth:
            deadrise_frac = 1 - T x tan(deadrise_deg) / (B/2)
        Clamped to [0.1, 0.95] to avoid degenerate hulls.
        """
        B_half_m = p.B / 2.0
        T_m = p.T

        deadrise_rad = np.radians(self.deadrise_deg)
        keel_reduction = T_m * np.tan(deadrise_rad) / B_half_m
        deadrise_frac = np.clip(1.0 - keel_reduction, 0.10, 0.95)

        taper_z = np.ones(nz, dtype=float)
        for k in range(nz):
            z_centre_m = (k + 0.5) * p.dz_m   # metres above keel (dz == vs in isotropic mode)
            if z_centre_m < T_m:
                # Below waterline: linear from deadrise_frac at keel to 1.0 at T
                t = z_centre_m / T_m                  # 0=keel, 1=waterline
                taper_z[k] = deadrise_frac + t * (1.0 - deadrise_frac)

        return taper_z


# ─────────────────────────────────────────────────────────────────
# Path B: STL-based hull mask
# ─────────────────────────────────────────────────────────────────

class STLHullMask:
    """
    Voxelises a ship hull STL mesh using trimesh to produce hull_mask[x, y, z].

    Workflow
    --------
    1. Load mesh (binary or ASCII STL) via trimesh.load()
    2. Validate watertightness — warn if non-manifold, continue regardless
    3. Align:
         - Translate bounding box minimum to origin (keel at z=0, aft at x=0)
         - Permute axes: longest ->x (longitudinal), second ->y (transverse),
           shortest ->z (vertical). Handles any export orientation.
    4. Scale: if fit_to_params=True, stretch/compress to match p.L x p.B x p.D
       exactly. This is the primary augmentation mechanism — one STL shape can
       represent many instances across the dimensional range for its ship type.
    5. Voxelise at pitch = p.voxel_size using trimesh's ray-casting voxeliser,
       then fill interior (trimesh.VoxelGrid.fill()).
    6. Extract (nx, ny, nz_hull) boolean matrix ->float32 hull_mask.

    Note on fractional boundary voxels
    ------------------------------------
    trimesh voxelisation returns binary occupancy (inside/outside), not fractional.
    Boundary voxels are therefore 0 or 1. This is acceptable at 3m resolution —
    the boundary layer is at most 1 voxel wide (~3m), which is well within the
    physical tolerance for early-stage arrangement generation.
    The parametric path retains fractional boundary handling if sub-voxel
    precision is needed for specific experiments.

    STL mesh requirements
    ----------------------
    - Watertight (closed manifold) preferred. Open meshes will produce a warning
      and trimesh will attempt voxelisation using the available surface.
    - Any scale — the aligner handles unit conversion automatically.
    - Any orientation — axis permutation is determined from bounding box extents.
    - Covers main hull only (keel to main deck). Superstructure is added
      programmatically by build_superstructure_mask() above the hull z-layers.
    """

    def __init__(self,
                 fit_to_params: bool = True,
                 use_fractional_boundary: bool = False,
                 assume_consistent_axes: bool = True):
        """
        Parameters
        ----------
        fit_to_params : bool
            True  (default) — scale mesh to match p.L x p.B x p.D.
                              Hull shape is preserved; only scale changes.
                              Enables one STL to generate instances across
                              the full dimensional range for its ship type.
            False           — keep mesh at its original scale; p dimensions
                              should match the STL file's actual metres.

        use_fractional_boundary : bool
            If True, compute a fractional hull_mask for STL meshes by
            sampling multiple points within each voxel using
            mesh.contains(...). This approximates the volume fraction of
            each voxel that lies inside the hull, similar in spirit to the
            parametric path. If False (default), use the faster binary
            voxelisation via trimesh.voxelized().fill().

        assume_consistent_axes : bool
            If True (default), assume all STLs already follow the project's
            axis convention (x=length, y=beam, z=up) and skip the extents-based
            axis permutation in `_align`. This avoids accidental y/z swaps when
            B and D are similar.
        """
        self.fit_to_params = fit_to_params
        self.use_fractional_boundary = use_fractional_boundary
        self.assume_consistent_axes = assume_consistent_axes

    def build(self, p: ShipParameterization,
              stl_path: str | Path) -> HullMaskResult:
        """
        Load, align, scale, and voxelise an STL mesh.

        Parameters
        ----------
        p        : ShipParameterization for this instance (from Stage 1)
        stl_path : path to .stl file

        Returns
        -------
        HullMaskResult with hull_mask (nx, ny, nz_hull) and ss_mask
        """
        if trimesh is None:
            raise ImportError(
                "trimesh is required for STL voxelisation. "
                "Install with: pip install trimesh"
            )
        stl_path = Path(stl_path)
        mesh = trimesh.load(str(stl_path), force='mesh')

        if not mesh.is_watertight:
            warnings.warn(
                f"STL mesh '{stl_path.name}' is not watertight. "
                f"Voxelisation may have interior gaps. "
                f"Consider repairing the mesh with trimesh.repair.fix_normals() "
                f"or MeshLab before use.",
                stacklevel=2
            )

        mesh = self._align(mesh, assume_consistent_axes=self.assume_consistent_axes)
        if self.fit_to_params:
            mesh = self._scale_to_params(mesh, p)

        if self.use_fractional_boundary:
            # mesh.contains() requires rtree; fall back to binary if unavailable
            try:
                import rtree  # noqa: F401
                hull_mask = self._voxelise_fractional(mesh, p)
            except ImportError:
                warnings.warn(
                    "rtree not installed — falling back to binary voxelisation. "
                    "Install with: pip install rtree",
                    stacklevel=2,
                )
                hull_mask = self._voxelise(mesh, p)
                # Binary voxelisation overestimates hull volume for slender
                # hulls (boundary voxels counted as fully occupied).  Use
                # mesh-based Cb instead of voxel-based for accuracy.
                self._compute_cb_from_mesh(mesh, p)
        else:
            hull_mask = self._voxelise(mesh, p)
            # Binary path: compute Cb from mesh geometry directly to avoid
            # the systematic overestimation of hull volume at coarse (3m)
            # voxel resolution.  See _compute_cb_from_mesh docstring.
            self._compute_cb_from_mesh(mesh, p)

        if self.use_fractional_boundary:
            # Fractional voxel grid tracks the mesh ground truth within
            # ~0.02 Cb, so the voxel-based estimate is physically accurate.
            denom = p.L * p.B * max(p.T, 1e-9)
            if denom > 0:
                T_v = p.T / p.dz_m
                z_idx = np.arange(p.nz_hull, dtype=np.float32)
                sub_frac = np.clip(T_v - z_idx, 0.0, 1.0).astype(np.float32)
                sub_vol_vox = float((hull_mask * sub_frac[None, None, :]).sum())
                cb_raw = (sub_vol_vox * p.cell_volume) / denom
                p.Cb = float(np.clip(cb_raw, 0.30, 0.95))
        # Surface is read directly from the STL geometry —
        # forecastle, bow sheer, etc. are all captured automatically.
        top_deck_z = compute_top_deck_z(hull_mask)
        full_mask, ss_only  = place_superstructure(hull_mask, top_deck_z, p)

        return HullMaskResult(
            full_mask=full_mask,
            top_deck_z=top_deck_z,
            params=p,
            hull_source=f"stl:{stl_path.name}",
            ss_only=ss_only,
            hull_fraction=hull_mask,
        )

    # ── Alignment ─────────────────────────────────────────────────────────────

    @staticmethod
    def _align(mesh: trimesh.Trimesh,
               assume_consistent_axes: bool = True) -> trimesh.Trimesh:
        """
        Translate and (optionally) permute axes so that:
          x = longitudinal (longest extent ->aft=0, bow=max)
          y = transverse   (second extent)
          z = vertical     (shortest extent ->keel=0, deck=max)

        For ships L >> B > D in general, so argsort(extents)[::-1] gives
        [longitudinal, transverse, vertical] in the original axes, and we
        permute to map them to [x, y, z].

        The permutation is applied as a 4x4 rotation matrix so trimesh
        correctly transforms both vertices and face normals.
        """
        mesh = mesh.copy()
        # Translate bounding box min to origin
        mesh.apply_translation(-mesh.bounds[0])

        if not assume_consistent_axes:
            extents = mesh.extents                          # [dx, dy, dz]
            order   = np.argsort(extents)[::-1]             # [longest, second, shortest]

            if not np.array_equal(order, [0, 1, 2]):
                # Build 3x3 permutation matrix P such that v_new = P @ v_old
                # v_new[i] = v_old[order[i]]  -> P[i, order[i]] = 1
                P = np.zeros((3, 3))
                for new_ax, old_ax in enumerate(order):
                    P[new_ax, old_ax] = 1.0
                T = np.eye(4)
                T[:3, :3] = P
                mesh.apply_transform(T)
                # Re-translate after permutation (origin may have shifted)
                mesh.apply_translation(-mesh.bounds[0])

        # At this point, the longest extent is along +x but we don't yet know
        # which end of the STL is bow vs aft. Mirror the x‑axis so that the
        # bow (typically at the "fine" end) ends up at positive x and the aft
        # at x=0, matching the parametric convention used elsewhere.
        T_flip = np.eye(4)
        T_flip[0, 0] = -1.0
        mesh.apply_transform(T_flip)
        # Re‑translate so that the new min-x is at the origin again
        mesh.apply_translation(-mesh.bounds[0])

        return mesh

    @staticmethod
    def _scale_to_params(mesh: trimesh.Trimesh,
                          p: ShipParameterization) -> trimesh.Trimesh:
        """
        Scale mesh so its bounding box matches p.L x p.B x p.D exactly.
        Non-uniform scaling is applied (hull shape is preserved only in
        the normalised sense; aspect ratios change to match the sampled
        dimensions). This is intentional — it is what allows a single
        STL shape to represent the full range of a ship type family.
        """
        mesh    = mesh.copy()
        extents = mesh.extents                          # [L_mesh, B_mesh, D_mesh]
        scale   = np.array([p.L, p.B, p.D]) / (extents + 1e-9)
        T       = np.diag([scale[0], scale[1], scale[2], 1.0])
        mesh.apply_transform(T)
        return mesh

    # ── Voxelisation ──────────────────────────────────────────────────────────

    @staticmethod
    def _voxelise(mesh: trimesh.Trimesh,
                   p: ShipParameterization) -> np.ndarray:
        """
        Voxelise mesh at p.voxel_size pitch and extract (nx, ny, nz_hull) mask.

        trimesh.voxelized() uses ray casting to determine interior/exterior.
        .fill() flood-fills the interior of the resulting voxel shell to ensure
        all inside voxels are marked, not just surface-adjacent ones.

        The resulting VoxelGrid.matrix is a boolean (mx, my, mz) array where
        True = inside hull. We crop or pad to exactly (nx, ny, nz_hull).
        """
        vg  = mesh.voxelized(pitch=p.voxel_size).fill()
        mat = vg.matrix.astype(np.float32)   # (mx, my, mz)
        mx, my, mz = mat.shape

        hull_mask = np.zeros((p.nx, p.ny, p.nz_hull), dtype=np.float32)

        # ── x axis: anchor at aft (x=0) ──────────────────────────────
        cx    = min(mx, p.nx)
        src_x = slice(0, cx)
        dst_x = slice(0, cx)

        # ── y axis: CENTRE in p.ny ────────────────────────────────────
        # trimesh.voxelized() can return my = p.ny ± 1 due to its internal
        # rounding, which shifts the hull to one transverse side if pasted
        # at y=0.  Centring absorbs that discrepancy and keeps the footprint
        # symmetric about y = ny/2, matching the parametric path and the
        # symmetry check in validate_hull_mask.
        if my >= p.ny:
            # mat wider than target — crop symmetrically from mat
            y_surplus = my - p.ny
            src_y = slice(y_surplus // 2, y_surplus // 2 + p.ny)
            dst_y = slice(0, p.ny)
        else:
            # mat narrower than target — pad symmetrically in target
            dy    = (p.ny - my) // 2
            src_y = slice(0, my)
            dst_y = slice(dy, dy + my)

        # ── z axis: anchor at keel (z=0) ─────────────────────────────
        cz    = min(mz, p.nz_hull)
        src_z = slice(0, cz)
        dst_z = slice(0, cz)

        hull_mask[dst_x, dst_y, dst_z] = mat[src_x, src_y, src_z]

        # ── Symmetry enforcement ──────────────────────────────────────
        # The physical hull is symmetric about the transverse centreline.
        # Trimesh voxelisation introduces a ±1 rounding that can produce
        # my = p.ny ± 1, leaving one y column unmirrored after centring.
        # Averaging hull_mask with its transverse mirror enforces exact
        # symmetry without distorting the hull form.
        hull_mask = (hull_mask + hull_mask[:, ::-1, :]) / 2.0

        return hull_mask

    # ── Fractional voxelisation (experimental) ──────────────────────────────
    @staticmethod
    def _voxelise_fractional(mesh: trimesh.Trimesh,
                             p: ShipParameterization,
                             samples_per_dim: int = 2) -> np.ndarray:
        """
        Approximate fractional hull occupancy per voxel by sampling points
        inside each voxel and querying mesh.contains(...).

        This is more expensive than the binary voxelisation in _voxelise, but
        yields hull_mask[x, y, z] ∈ [0, 1] that better captures partial
        boundary cells, similar in spirit to the parametric path.

        Coordinate convention after _align and _scale_to_params:
            x ∈ [0, L], y ∈ [0, B], z ∈ [0, D]
        Voxel i,j,k spans:
            x ∈ [i*v, (i+1)*v],   y ∈ [j*v, (j+1)*v],   z ∈ [k*v, (k+1)*v]
        where v = p.voxel_size.
        """
        nx, ny, nz = p.nx, p.ny, p.nz_hull
        dx = p.dx_m; dy = p.dy_m; dz = p.dz_m
        v = p.voxel_size

        hull_mask = np.zeros((nx, ny, nz), dtype=np.float32)

        # Sub-voxel sample offsets in [0, 1) scaled by voxel_size.
        # For samples_per_dim=2 this yields 8 points at (0.25, 0.75)^3.
        coords_1d = (np.arange(samples_per_dim) + 0.5) / samples_per_dim
        gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d,
                                 indexing="ij")
        sub_offsets = np.stack(
            [gx.reshape(-1)*dx, gy.reshape(-1)*dy, gz.reshape(-1)*dz], axis=-1)

        # Loop over voxels; for typical grids (nx~40-80, ny~10-25, nz~6-10)
        # this is manageable for experimentation.
        for i in range(nx):
            x_base = i * dx
            for j in range(ny):
                y_base = j * dy
                for k in range(nz):
                    z_base = k * dz
                    base = np.array([x_base, y_base, z_base], dtype=float)
                    pts = base + sub_offsets  # (n_samples, 3)
                    inside = mesh.contains(pts)
                    hull_mask[i, j, k] = float(np.mean(inside))

        # Enforce transverse symmetry, as in the binary path.
        hull_mask = (hull_mask + hull_mask[:, ::-1, :]) / 2.0
        return hull_mask

    # ── Mesh-based Cb (for binary voxelisation path) ──────────────────────

    @staticmethod
    def _compute_cb_from_mesh(mesh: trimesh.Trimesh,
                               p: ShipParameterization,
                               n_samples: int = 5000) -> None:
        """
        Compute block coefficient directly from the mesh geometry by
        sampling random points in the submerged box [0,L]×[0,B]×[0,T]
        and querying ``mesh.contains()``.

        This avoids the systematic Cb overestimation that occurs with
        binary voxelisation at coarse (3m) resolution, where boundary
        voxels are counted as fully occupied.  Tested to within ~0.02
        of the fractional-voxel Cb (which itself matches dense-sampling
        ground truth).

        Mutates ``p.Cb`` in place.  Requires rtree (via mesh.contains).
        If rtree is unavailable, falls back to the voxel-based estimate
        via the standard formula and logs a warning.
        """
        L, B, T = p.L, p.B, p.T
        denom = L * B * max(T, 1e-9)
        if denom <= 0:
            return

        try:
            rng = np.random.default_rng(0)
            pts = rng.random((n_samples, 3))
            pts[:, 0] *= L
            pts[:, 1] *= B
            pts[:, 2] *= T
            inside = mesh.contains(pts)
            cb_raw = float(inside.mean())
            p.Cb = float(np.clip(cb_raw, 0.30, 0.95))
        except Exception:
            # rtree unavailable or mesh.contains failed — fall back to
            # voxel-based estimate (less accurate for binary, but safe).
            warnings.warn(
                "mesh.contains() failed for Cb computation; "
                "using voxel-based estimate (may overestimate for slender hulls).",
                stacklevel=3,
            )


# ─────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────

def hull_bbox_fill_fraction(result: HullMaskResult) -> float:
    """
    Hull “fill fraction” used by QC: occupied hull voxel units divided by
    the axis-aligned hull grid volume (nx × ny × nz_hull).

    Each voxel contributes its *occupancy* (0–1 for fractional STL
    boundaries, or 0/1 for binary masks).  The denominator is the full
    rectangular box from keel to main-deck height — not displaced volume.

    Interpretation
    ----------------
    - **High (~0.7–0.9):** full midship section, blocky parametric forms.
    - **Mid (~0.5–0.65):** typical cargo forms with taper.
    - **Low (<0.5):** slender hulls, strong bow/stern taper, or an STL
      scaled inside a loose L×B×D box at coarse (e.g. 3 m) voxel size —
      often trips the default QC band [0.50, 0.95].

    Same definition as ``validate_hull_mask`` check #3.
    """
    p = result.params
    denom = float(p.nx * p.ny * p.nz_hull) + 1e-6
    return float(result.hull_volume_voxels) / denom


def format_hull_qc_reject_log(
    ship_type_name: str,
    result: HullMaskResult,
    warns: list,
) -> str:
    """Multi-line message for logging when ``validate_hull_mask`` fails."""
    fill = hull_bbox_fill_fraction(result)
    p = result.params
    bbox_cells = p.nx * p.ny * p.nz_hull
    occ = result.hull_volume_voxels
    lines = [
        "",
        "[HullMask QC rejected]",
        f"  ship_type    : {ship_type_name}",
        f"  hull_source  : {result.hull_source}",
        f"  voxel_grid   : nx={p.nx}  ny={p.ny}  nz_hull={p.nz_hull}  "
        f"voxel={p.voxel_size}m",
        f"  bbox_cells   : {bbox_cells}  (hull box only, no SS layers)",
        f"  occupied_sum : {occ:.1f}  voxel occupancy units",
        f"  fill_fraction: {fill:.4f}  (= occupied_sum / bbox_cells)",
        f"  hull_dims    : L={p.L:.1f}m  B={p.B:.1f}m  D={p.D:.1f}m  "
        f"T={p.T:.1f}m  Cb={p.Cb:.3f}",
        f"  qc_note      : fill is NOT Cb; it measures how much of the "
        f"LxBxD grid is filled.",
        f"  warnings     :",
    ]
    for w in warns:
        lines.append(f"    - {w}")
    lines.append("")
    return "\n".join(lines)


def validate_hull_mask(result: HullMaskResult) -> Tuple[bool, list]:
    """
    Sanity checks on a generated hull mask.

    Checks
    ------
    1. No NaN or Inf values
    2. All values approximately in [0, 1]
    3. Hull bounding-box fill fraction in an expected band (binary vs
       fractional voxelisation use different lower bounds; see implementation)
    4. Transverse symmetry: hull_mask[:, y, :] ≈ hull_mask[:, ny-1-y, :]
    5. Double-bottom layer occupancy vs overall hull fill (scaled threshold;
       weak keel fill is allowed for V-bottom forms)
    """
    p = result.params
    mask = result.hull_mask
    warns = []

    if np.isnan(mask).any() or np.isinf(mask).any():
        warns.append("Hull mask contains NaN or Inf values")
    if mask.min() < -0.01 or mask.max() > 1.01:
        warns.append(f"Hull mask out of [0,1] range: [{mask.min():.3f}, {mask.max():.3f}]")

    # Volume check: fill fraction = hull voxels / bounding-box voxels
    # Binary voxelisation: typical range 0.55-0.92 for normal ship forms.
    # Fractional voxelisation: boundary voxels are 0.1-0.9 instead of 1.0,
    # so fine/slender hulls can drop to 0.30-0.45 legitimately.
    # Detect fractional masks by checking for non-binary values.
    fill = hull_bbox_fill_fraction(result)
    has_fractional = bool(np.any((mask > 0.01) & (mask < 0.99)))
    fill_lo = 0.25 if has_fractional else 0.50
    if not (fill_lo <= fill <= 0.95):
        warns.append(f"Hull fill fraction = {fill:.3f}, expected [{fill_lo:.2f}, 0.95]")

    # Symmetry check
    ny = p.ny
    if ny >= 4:
        sym_err = np.abs(mask - mask[:, ::-1, :]).mean()
        if sym_err > 0.05:
            warns.append(f"Transverse symmetry error = {sym_err:.4f} (threshold 0.05)")

    # Double-bottom occupancy
    overall_fill = float(mask.sum()) / max(1, p.nx * p.ny * p.nz_hull)
    db_fill = mask[:, :, :p.db_layers].sum() / max(1, p.nx * p.ny * p.db_layers)

    # Guard against a *degenerate* bottom (voxelisation emptied the keel band),
    # NOT a merchant-fullness requirement. Fine-keeled hulls (warships, V-bottom
    # yachts) legitimately fill only a few % of the bottom band, and the band
    # sits lower/thinner as the grid refines. Recalibrate against real hulls at
    # the production grid (esp. the real yacht); until a real patrol is encoded
    # this floor is geometric, not data-anchored.
    db_floor = max(0.008, 0.025 * overall_fill) 
    if db_fill < db_floor:
        warns.append(f"Double bottom region near-empty (fill={db_fill:.3f} "
                     f"< {db_floor:.3f}) — hull may be degenerate at this grid")

    return len(warns) == 0, warns
