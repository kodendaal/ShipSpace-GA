"""Read-only engineering assessment of *existing* native semantic arrangements.

Uses unmodified P1A weight/hydro/adjacency kernels, independent of generator.
Inputs must be native x,y,z arrays with fractional occupancy, NOT compact voxels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import hashlib
import math

import numpy as np

from ._p1a_weight.config import load_config, family_config
from ._p1a_weight.hydrostatics import integrate_hydrostatics
from ._p1a_weight.weights import build_weight_model
from ._p1a_weight.deck_guard import extract_local_deck_envelope, required_freeboard_m
from ._p1a_weight.solver import solve_heave_and_trim
from ._p1a_weight.free_surface import estimate_free_surface
from ._p1a_adjacency.core import VolumeGeometry, compute_functional_adjacency, EMPTY

CALIBRATION = Path(__file__).resolve().parent / 'calibration' / 'family_residual_closure_v4_production_calibrated_fsm_shared.json'
FAMILIES = ('BULKER', 'TANKER', 'CARGO', 'OSV', 'PATROL', 'YACHT')


def _array(x: Any) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _meta(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "files"):
        return obj[key] if key in obj.files else default
    return getattr(obj, key, default)


@dataclass(frozen=True)
class EngineeringVolume:
    """Native 3-D labels + fractional cell availability.

    labels: 0..8 in-hull, 10 outside; 9 NAVIGATION is forbidden.
    hull_fraction: 0..1 full native grid incl. SS (fraction=1 for full SS cells).
    main_hull_fraction: optional *original* hull-only mask for P1A hydrostatics.
    If absent, the main-hull slice of hull_fraction is used instead; this can
    differ from the source hull when zones are excised during construction.
    """
    labels: np.ndarray
    hull_fraction: np.ndarray
    family: str
    L: float
    B: float
    D: float
    nz_hull: int
    T: float | None = None
    main_hull_fraction: np.ndarray | None = None
    sample_id: str = ''
    mask_provenance: str = 'unspecified'

    def validate(self) -> None:
        labels = _array(self.labels)
        frac = _array(self.hull_fraction)
        if labels.ndim != 3 or frac.shape != labels.shape:
            raise ValueError('labels and hull_fraction must have identical native (nx,ny,nz) dimensions')
        if labels.size == 0 or min(labels.shape) == 0:
            raise ValueError('empty volume')
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('labels must use integer semantic indices, not probabilities')
        if (labels < 0).any() or ((labels > 8) & (labels != EMPTY)).any():
            raise ValueError('only semantic labels 0..8 and EMPTY=10 are supported (NAVIGATION=9 forbidden)')
        if not np.isfinite(frac).all() or np.any(frac < 0) or np.any(frac > 1+1e-6):
            raise ValueError('hull fractions must be finite within [0,1]')
        if np.any((labels != EMPTY) & (frac <= 0)) or np.any((labels == EMPTY) & (frac > 0)):
            raise ValueError('EMPTY cells must have zero occupancy; non-EMPTY cells must have positive occupancy')
        if self.family.upper() not in FAMILIES:
            raise ValueError(f'unknown family {self.family!r}')
        if not all(math.isfinite(float(x)) and float(x)>0 for x in (self.L,self.B,self.D)):
            raise ValueError('L B D must be finite, positive metres')
        if not 1 <= int(self.nz_hull) <= labels.shape[2]:
            raise ValueError('nz_hull must be within native z extent')
        if self.main_hull_fraction is not None:
            mh = _array(self.main_hull_fraction)
            if mh.shape != (labels.shape[0],labels.shape[1],int(self.nz_hull)):
                raise ValueError('main_hull_fraction has wrong shape')
            if not np.isfinite(mh).all() or np.any(mh < 0) or np.any(mh > 1+1e-6):
                raise ValueError('main_hull_fraction outside [0,1]')
        if self.T is not None and not (0 < float(self.T) <= float(self.D)):
            raise ValueError('T must be positive and <=D')


def _native_model_helpers(vol: EngineeringVolume):
    # Reuse exact current ShipSpace-GA enum and density_for (incl. yacht rule).
    from src.data_generator.ship_params import Comp, ShipType, density_for
    labels = _array(vol.labels).astype(np.int16,copy=False)
    alpha = _array(vol.hull_fraction).astype(np.float64,copy=False)
    nx,ny,nz = labels.shape
    dx,dy,dz = float(vol.L)/nx,float(vol.B)/ny,float(vol.D)/int(vol.nz_hull)
    ship_type = ShipType[vol.family.upper()]
    p = SimpleNamespace(nx=nx,ny=ny,nz_hull=int(vol.nz_hull),nz_total=nz,
                        L=float(vol.L),B=float(vol.B),D=float(vol.D),
                        T=float(vol.T) if vol.T is not None else 0.7*float(vol.D),
                        dx_m=dx,dy_m=dy,dz_m=dz,ship_type=ship_type,
                        displacement=0.0)
    x=np.broadcast_to((np.arange(nx)+0.5)[:,None,None]*dx,labels.shape)
    z=np.broadcast_to((np.arange(nz)+0.5)[None,None,:]*dz,labels.shape)
    fields=SimpleNamespace(labels=labels,volume_m3=alpha*dx*dy*dz,x_m=x,z_m=z)
    a=SimpleNamespace(layout=SimpleNamespace(params=p),actual_lcg_frac=0.5,actual_kg_frac=0.5)
    api=SimpleNamespace(Comp=Comp,density_for=density_for,build_voxel_fields_from_assignment=lambda _:fields)
    return p,a,api,fields


def _adjacency(vol: EngineeringVolume) -> dict[str,Any]:
    lab=_array(vol.labels).astype(np.int16,copy=False)
    al=_array(vol.hull_fraction).astype(float,copy=False)
    nx,ny,nz=lab.shape
    dx=float(vol.L)/nx; dy=float(vol.B)/ny; dz=float(vol.D)/vol.nz_hull
    ix=np.arange(nx)[:,None,None];iy=np.arange(ny)[None,:,None];iz=np.arange(nz)[None,None,:]
    x0=np.broadcast_to(ix*dx,lab.shape); y0=np.broadcast_to(iy*dy,lab.shape);z0=np.broadcast_to(iz*dz,lab.shape)
    row,_=compute_functional_adjacency(VolumeGeometry(
        labels=lab,volume_m3=al*dx*dy*dz,x0=x0,x1=x0+dx,y0=y0,y1=y0+dy,
        z0=z0,z1=z0+dz,L=float(vol.L),B=float(vol.B),D=float(vol.D)),
        void_traversable=True,ship_id=vol.sample_id or 'unidentified',source='frozen_assessment',family=vol.family.upper())
    return row


def evaluate_volume(vol: EngineeringVolume, *, equilibrium: bool=False,
                    config_path: str | Path | None=None, adjacency: bool=True) -> dict[str,Any]:
    """Evaluate P1A original kernels on a saved arrangement, without regeneration.

    By default calculates LCG/KG and J_adj. ``equilibrium=True`` additionally
    runs heave/trim and FSM-corrected GM (slower, calibration-dependent).
    Invalid weights/adjacency and nonconvergence are returned with explicit status.
    """
    vol.validate()
    cfg_path=Path(config_path) if config_path is not None else CALIBRATION
    cfg=load_config(cfg_path); fcfg=family_config(cfg,vol.family.upper())
    p,a,api,_ = _native_model_helpers(vol)
    hull = (_array(vol.main_hull_fraction) if vol.main_hull_fraction is not None else _array(vol.hull_fraction)[:,:,:vol.nz_hull]).astype(float,copy=False)
    rho=float(fcfg.get('rho_sw_t_m3',1.025))
    mode=str(fcfg.get('reference_draft_mode','sampled_draft'))
    if mode=='fixed_T_D': draft=float(fcfg['reference_T_D'])*p.D
    elif mode=='fixed_draft_m':draft=float(fcfg['reference_draft_m'])
    elif mode=='sampled_draft':draft=p.T
    elif mode=='local_deck_safe_fraction':
        env=extract_local_deck_envelope(hull,p,occupancy_threshold=float(fcfg.get('deck_occupancy_threshold',0.1)),
           central_beam_fraction=float(fcfg.get('deck_central_beam_fraction',0.6)),
           station_quantile=float(fcfg.get('deck_station_quantile',0.25)),
           end_exclusion_fraction=float(fcfg.get('deck_end_exclusion_fraction',0.03)),
           min_station_breadth_fraction=float(fcfg.get('deck_min_station_breadth_fraction',0.18)),
           smoothing_stations=int(fcfg.get('deck_smoothing_stations',3)),
           step_threshold_m=fcfg.get('deck_step_threshold_m'),source=vol.mask_provenance)
        if not np.any(env.valid): raise ValueError('no valid deck envelope')
        draft=(np.min(env.deck_z_m[env.valid]-required_freeboard_m(p,fcfg))*float(fcfg.get('reference_safe_draft_fraction',0.90)))
    else: raise ValueError(f'unsupported reference_draft_mode {mode}')
    draft=float(np.clip(draft,1e-5,p.nz_hull*p.dz_m-1e-5))
    hydro=integrate_hydrostatics(hull,p,draft,0.0,rho)
    weight=build_weight_model(a,p,api,fcfg,reference_hydro=hydro)
    res:dict[str,Any]={
        'sample_id':vol.sample_id,'family':vol.family.upper(),
        'calibration_sha256':hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        'mask_provenance':vol.mask_provenance,
        'main_hull_source':'original_main_hull_fraction' if vol.main_hull_fraction is not None else 'from_native_occupied_fraction',
        'weight_valid':bool(weight.mass_model_valid),
        'weight_invalid_reason':weight.invalid_reason,
        'LCG_m':float(weight.lcg_m) if weight.mass_model_valid else float('nan'),
        'KG_m':float(weight.kg_m) if weight.mass_model_valid else float('nan'),
        'LCG_over_L':float(weight.lcg_m/p.L) if weight.mass_model_valid else float('nan'),
        'KG_over_D':float(weight.kg_m/p.D) if weight.mass_model_valid else float('nan'),
        'compartment_LCG_m':float(weight.compartment_lcg_m),
        'compartment_KG_m':float(weight.compartment_kg_m),
        'compartment_mass_t':float(weight.compartment_t),
        'total_mass_t':float(weight.total_t) if weight.mass_model_valid else float('nan'),
        'residual_mass_t':float(weight.residual_t),
        'reference_displacement_t':float(hydro.displacement_t),
        'reference_draft_m':float(hydro.mean_draft_m),
        'J_adj':float('nan'),'adjacency_valid':False,'adjacency_invalid_reason':'not_requested',
        'trim_status':'not_requested','trim_m':float('nan'),'trim_pct_L':float('nan'),
        'J_trim_abs_over_L':float('nan'),'GM_corrected_m':float('nan'),
        'equilibrium_valid':False,
    }
    if adjacency:
        jr=_adjacency(vol)
        res.update({
            'J_adj':float(jr['J_adj']), 'adjacency_valid':bool(jr['all_services_valid']),
            'adjacency_invalid_reason':jr['invalid_reason'],
            'n_adj_nodes':int(jr['n_nodes']), 'n_adj_edges':int(jr['n_edges']),
            'n_adj_required_pairs':int(jr['n_required_pairs']),
        })
        res.update({k:float(jr[k]) for k in ('D_fuel_ER','D_machinery_ER','D_accommodation_ER','D_steering_ER','D_accommodation_machinery')})
    if equilibrium:
        if not weight.mass_model_valid:
            res['trim_status']='mass_model_invalid'
        else:
            env=extract_local_deck_envelope(hull,p,occupancy_threshold=float(fcfg.get('deck_occupancy_threshold',0.1)),
                central_beam_fraction=float(fcfg.get('deck_central_beam_fraction',0.6)),
                station_quantile=float(fcfg.get('deck_station_quantile',0.25)),
                end_exclusion_fraction=float(fcfg.get('deck_end_exclusion_fraction',0.0)),
                min_station_breadth_fraction=float(fcfg.get('deck_min_station_breadth_fraction',0.05)),
                smoothing_stations=int(fcfg.get('deck_smoothing_stations',3)),
                step_threshold_m=fcfg.get('deck_step_threshold_m'),source=vol.mask_provenance)
            eq=solve_heave_and_trim(hull,p,weight,env,
                required_freeboard_m=required_freeboard_m(p,fcfg),rho_sw_t_m3=rho,
                trim_limit_L=float(fcfg.get('trim_limit_L',0.04)),
                trim_grid_points=int(fcfg.get('trim_grid_points',33)),
                weight_tolerance_fraction=float(fcfg.get('weight_tolerance_fraction',1e-6)),
                lcb_tolerance_L=float(fcfg.get('lcb_tolerance_L',0.002)))
            res['trim_status']=eq.status
            if eq.hydro is not None:
                h=eq.hydro
                fsc=estimate_free_surface(a,api,fcfg,h.displacement_t)
                tr=float(h.trim_m)
                res.update({'trim_m':tr,'trim_pct_L':100*tr/p.L,
                    'J_trim_abs_over_L':abs(tr)/p.L,
                    'GM_corrected_m':float(h.km_transverse_m-weight.kg_m-fsc.correction_m),
                    'equilibrium_valid':bool(eq.converged),
                    'free_surface_correction_m':float(fsc.correction_m),
                    'equilibrium_draft_m':float(h.mean_draft_m),
                    'equilibrium_displacement_t':float(h.displacement_t),
                    'LCB_m':float(h.lcb_m)})
    return res


def from_companion(graph: Any, companion: Any, *, labels_by_node: Any | None=None,
                   main_hull_fraction: Any | None=None, sample_id: str='') -> EngineeringVolume:
    """Reconstruct native arrangement using frozen companion's native_zone_id map.

    labels_by_node can be an alternative node-label vector (n_nodes,).
    If None, uses companion.native_zone_labels (reference). This does not
    decode compact label grids and therefore preserves native zone geometry.
    """
    node=_array(_meta(companion,'native_zone_id'))
    alpha=_array(_meta(companion,'native_occupied_fraction'))
    if node.ndim!=3 or alpha.shape!=node.shape:
        raise ValueError('companion needs native_zone_id and native_occupied_fraction')
    y=_array(_meta(companion,'native_zone_labels') if labels_by_node is None else labels_by_node)
    if not np.issubdtype(node.dtype, np.integer) or np.any(node < -1):
        raise ValueError('native_zone_id must contain integer node indices or -1 outside')
    if y.ndim!=1 or y.size <= int(node.max(initial=-1)):
        raise ValueError('labels_by_node length does not match node indices')
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y > 8)):
        raise ValueError('labels_by_node must contain integer semantic labels 0..8')
    labels=np.full(node.shape,EMPTY,dtype=np.int16)
    inside=node>=0
    labels[inside]=y[node[inside]].astype(np.int16)
    labels[(alpha<=0)]=EMPTY
    fam=_meta(graph,'ship_type')
    if fam is None:raise ValueError('graph missing ship_type')
    try:
        from src.data_generator.ship_params import ShipType
        family=ShipType(int(fam)).name
    except Exception as e:raise ValueError(f'unknown ship_type={fam}') from e
    return EngineeringVolume(labels=labels,hull_fraction=alpha,family=family,
        L=float(_meta(graph,'L')),B=float(_meta(graph,'B')),D=float(_meta(graph,'D')),
        nz_hull=int(_meta(graph,'nz_hull')),T=float(_meta(graph,'T')),
        main_hull_fraction=None if main_hull_fraction is None else _array(main_hull_fraction),
        sample_id=sample_id,mask_provenance='native_companion')


def compare_pair(reference: EngineeringVolume, prediction: EngineeringVolume,
                 *, equilibrium: bool=False,config_path: str | Path | None=None) -> dict[str,Any]:
    """Enforce identical physical envelope; calculate both and paired changes."""
    reference.validate();prediction.validate()
    if (reference.family.upper(),reference.L,reference.B,reference.D,reference.nz_hull,reference.T) != (prediction.family.upper(),prediction.L,prediction.B,prediction.D,prediction.nz_hull,prediction.T):
        raise ValueError('reference/prediction ship particulars differ')
    if not np.array_equal(reference.hull_fraction,prediction.hull_fraction):
        raise ValueError('reference/prediction hull occupation differs')
    if (reference.main_hull_fraction is None)!=(prediction.main_hull_fraction is None):
        raise ValueError('reference/prediction main-hull provenance differs')
    if reference.main_hull_fraction is not None and not np.array_equal(reference.main_hull_fraction,prediction.main_hull_fraction):
        raise ValueError('reference/prediction main hull differs')
    r=evaluate_volume(reference,equilibrium=equilibrium,config_path=config_path)
    q=evaluate_volume(prediction,equilibrium=equilibrium,config_path=config_path)
    out={'sample_id':prediction.sample_id or reference.sample_id,
         'reference':r,'prediction':q,
         'delta_LCG_over_L':q['LCG_over_L']-r['LCG_over_L'],
         'delta_KG_over_D':q['KG_over_D']-r['KG_over_D'],
         'delta_J_adj':q['J_adj']-r['J_adj'] if q['adjacency_valid'] and r['adjacency_valid'] else float('nan'),
         'pair_adjacency_valid':bool(q['adjacency_valid'] and r['adjacency_valid'])}
    return out


def from_live_assignment(assignment: Any, *, sample_id: str='') -> EngineeringVolume:
    """Read current generator CompartmentAssignment without invoking generator.

    Intended for an already-generated assignment; this neither changes the
    assignment nor invokes Stage-4. Prefer this for P1A-equivalent hull sources.
    """
    layout=assignment.layout; p=layout.params; hr=layout.hull_result
    zone=_array(layout.zone_mask)
    frac=_array(hr.full_mask).astype(float,copy=True)
    if zone.shape!=frac.shape:raise ValueError('live zone mask and full mask differ')
    labels=np.full(zone.shape,EMPTY,dtype=np.int16)
    for zid, comp in assignment.assignments.items():
        labels[zone==int(zid)]=int(comp)
    frac[labels==EMPTY]=0.0
    hull=_meta(hr,'hull_mask',None)
    if hull is None:hull=_meta(hr,'hull_fraction',None)
    if hull is None:raise ValueError('live assignment missing original main hull fraction')
    main=_array(hull).astype(float,copy=False)[:,:,:int(p.nz_hull)]
    return EngineeringVolume(labels=labels,hull_fraction=frac,
        family=p.ship_type.name,L=float(p.L),B=float(p.B),D=float(p.D),
        nz_hull=int(p.nz_hull),T=float(p.T),main_hull_fraction=main,
        sample_id=sample_id,mask_provenance='generator_hull_result')
