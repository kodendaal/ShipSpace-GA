"""
stl_mask_cache.py — fractional STL hull voxelisations, stored once per hull form.

Why caching is exact
--------------------
`STLHullMask._scale_to_params` stretches the mesh per-axis to fill the
[0,L]x[0,B]x[0,D] box, and `_voxelise_fractional` steps by dx=L/Nx (etc.) over
that same box. Every sample point therefore lands at a fixed *normalised*
fraction of the mesh, so the L/B/D cancel: for a given (hull form, grid_counts)
the index-space fractional hull mask is identical regardless of the sampled
principal dimensions. (Verified: two ships 182x32x13 and 205x36x22, same STL +
grid, give byte-identical masks.)

The voxelisation is therefore stored once per (hull form, grid_counts) as
``<form>_<nx>x<ny>x<nz>_<digest>.npy`` in ``data/hull_cache`` and reused for
every sample. The stored grids are the hull library: the STL meshes are only
needed to add a new hull form. The sampled dimensions re-enter only as the
scalars dx/dy/dz/cell_volume (Cb from the cached mask) and the per-sample
superstructure stack + zone layout, all of which are cheap.

This is only valid for the FRACTIONAL path (rtree present). The binary
fallback voxelises with a cubic pitch on the scaled mesh and is NOT
dimension-invariant, so caching is refused there.

Usage
-----
    from stl_mask_cache import CachedSTLHullMask

    hb = CachedSTLHullMask(fit_to_params=True, use_fractional_boundary=True,
                           cache_dir=HULL_CACHE_DIR)
    hr = hb.build(p, stl_path=...)
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None

from hull_mask import (
    HULL_CACHE_DIR, STLHullMask, HullMaskResult, compute_top_deck_z, place_superstructure,
)

PRODUCTION_GRID: Tuple[int, int, int] = (64, 32, 24)
DEFAULT_HULL_CACHE_DIR = HULL_CACHE_DIR


def _rtree_available() -> bool:
    try:
        import rtree  # noqa: F401
        return True
    except ImportError:
        return False


class CachedSTLHullMask(STLHullMask):
    """STLHullMask whose fractional hull voxelisation is read from the hull
    cache by hull-form name and grid. Behaviourally identical to STLHullMask
    on the fractional path; only the expensive mesh.contains() step is
    replaced by the stored grid."""

    # process-wide in-memory cache shared across instances
    _MEM: Dict[Tuple[str, Tuple[int, int, int]], np.ndarray] = {}

    def __init__(self, *args, cache_dir: Optional[str | Path] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.use_fractional_boundary:
            raise ValueError(
                "CachedSTLHullMask requires use_fractional_boundary=True; the "
                "binary voxeliser is not dimension-invariant and cannot be cached."
            )
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    # ── cache lookup ─────────────────────────────────────────────────────
    @staticmethod
    def _grid(p) -> Tuple[int, int, int]:
        # per-axis hull cell counts (the mask shape) live on p as nx/ny/nz_hull
        if not all(hasattr(p, a) for a in ("nx", "ny", "nz_hull")):
            raise ValueError("CachedSTLHullMask requires an anisotropic grid "
                             "(p.nx, p.ny, p.nz_hull set).")
        return (int(p.nx), int(p.ny), int(p.nz_hull))

    def _cache_file(self, stl_path: Path, p) -> Optional[Path]:
        """Stored voxelisation of this hull form at this grid, if present."""
        if self.cache_dir is None:
            return None
        nx, ny, nz = self._grid(p)
        stem = Path(stl_path).stem
        matches = sorted(self.cache_dir.glob(f"{stem}_{nx}x{ny}x{nz}_*.npy"))
        if len(matches) > 1:
            raise RuntimeError(
                f"Several stored masks for '{stem}' at {nx}x{ny}x{nz} in {self.cache_dir}"
            )
        return matches[0] if matches else None

    # ── the only expensive step, memoised ────────────────────────────────
    def _hull_mask(self, stl_path: Path, p) -> np.ndarray:
        stl_path = Path(stl_path)
        key = (stl_path.stem, self._grid(p))
        if key in self._MEM:
            return self._MEM[key]

        stored = self._cache_file(stl_path, p)
        if stored is not None:
            hm = np.load(stored).astype(np.float32)
            self._MEM[key] = hm
            return hm

        if not stl_path.is_file():
            nx, ny, nz = key[1]
            raise FileNotFoundError(
                f"No stored hull mask for '{stl_path.stem}' at {nx}x{ny}x{nz} "
                f"in {self.cache_dir} and no STL mesh at {stl_path}"
            )
        if not _rtree_available():
            raise RuntimeError(
                "rtree is required for the cacheable fractional voxeliser. "
                "Install with: pip install rtree"
            )
        if trimesh is None:
            raise ImportError("trimesh is required for STL voxelisation.")

        # full expensive path: load -> align -> scale -> fractional voxelise
        mesh = trimesh.load(str(stl_path), force="mesh")
        mesh = self._align(mesh, assume_consistent_axes=self.assume_consistent_axes)
        if self.fit_to_params:
            mesh = self._scale_to_params(mesh, p)
        hm = self._voxelise_fractional(mesh, p).astype(np.float32)

        self._MEM[key] = hm
        if self.cache_dir is not None:
            nx, ny, nz = key[1]
            digest = hashlib.sha256(hm.tobytes()).hexdigest()[:8]
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{stl_path.stem}_{nx}x{ny}x{nz}_{digest}.npy", hm)
        return hm

    # ── per-sample finalise (mirror of STLHullMask.build tail) ────────────
    def build(self, p, stl_path: str | Path) -> HullMaskResult:
        stl_path = Path(stl_path)
        hull_mask = self._hull_mask(stl_path, p)   # cached; dimension-invariant

        # Cb from the fractional hull mask + this sample's draft/scale.
        denom = p.L * p.B * max(p.T, 1e-9)
        if denom > 0:
            T_v = p.T / p.dz_m
            z_idx = np.arange(p.nz_hull, dtype=np.float32)
            sub = np.clip(T_v - z_idx, 0.0, 1.0).astype(np.float32)
            sub_vol_vox = float((hull_mask * sub[None, None, :]).sum())
            p.Cb = float(np.clip((sub_vol_vox * p.cell_volume) / denom, 0.30, 0.95))

        top_deck_z = compute_top_deck_z(hull_mask)
        full_mask, ss_only = place_superstructure(hull_mask, top_deck_z, p)
        return HullMaskResult(
            full_mask=full_mask,
            top_deck_z=top_deck_z,
            params=p,
            hull_source=f"stl:{stl_path.name}",
            ss_only=ss_only,
            hull_fraction=hull_mask,
        )


def precompute_cache(stl_library: Dict[object, Iterable[str | Path]],
                     grid_counts: Tuple[int, int, int],
                     cache_dir: str | Path,
                     sampler=None) -> int:
    """Voxelise every hull form in `stl_library` that has no stored mask at
    `grid_counts`. Run once (single process) before launching parallel
    workers. Returns the number of masks written.

    `stl_library`: {ship_type: [stl_path, ...]}.
    `sampler`: optional ShipParameterizationSampler(grid_counts=...); if None a
    minimal one is constructed per type.
    """
    from ship_params import ShipParameterizationSampler
    hb = CachedSTLHullMask(fit_to_params=True, use_fractional_boundary=True,
                           cache_dir=cache_dir)
    n = 0
    for ship_type, paths in stl_library.items():
        smp = sampler or ShipParameterizationSampler(grid_counts=grid_counts)
        p = smp.sample(ship_type)            # any dims — mask is invariant
        for sp in paths:
            sp = Path(sp)
            if (sp.stem, hb._grid(p)) in hb._MEM or hb._cache_file(sp, p) is not None:
                continue
            hb._hull_mask(sp, p)
            n += 1
    return n


def cache_populated(cache_dir: Path) -> bool:
    return cache_dir.is_dir() and any(cache_dir.glob("*.npy"))


def resolve_hull_cache_dir(cache_dir: Optional[str | Path] = None) -> Path:
    return Path(cache_dir if cache_dir is not None else DEFAULT_HULL_CACHE_DIR).resolve()


def make_stl_builder(cache_dir: Optional[Path] = None):
    """Return CachedSTLHullMask when cache_dir is set, else raw STLHullMask."""
    if cache_dir is not None:
        return CachedSTLHullMask(
            fit_to_params=True,
            use_fractional_boundary=True,
            cache_dir=cache_dir,
        )
    return STLHullMask(fit_to_params=True, use_fractional_boundary=True)


def ensure_hull_cache(
    stl_library,
    *,
    cache_dir: Optional[str | Path] = None,
    grid_counts: Tuple[int, int, int] = PRODUCTION_GRID,
) -> int:
    """Add stored masks for hull forms that have none (never deletes entries)."""
    resolved = resolve_hull_cache_dir(cache_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    n = precompute_cache(
        stl_library, grid_counts=grid_counts, cache_dir=resolved,
    )
    total = len(list(resolved.glob("*.npy")))
    if n:
        print(
            f"Hull cache updated: {resolved} ({n} new, {total} total)",
            flush=True,
        )
    else:
        print(f"Hull cache up to date: {resolved} ({total} masks)", flush=True)
    return n
