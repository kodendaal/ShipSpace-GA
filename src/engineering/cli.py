"""File-based P1A engineering evaluation of existing native arrangements.

    python -m src.engineering.cli --input some_case.npz --output metrics.jsonl
    python -m src.engineering.cli --input path/to/npzdir --output metrics.csv

Required NPZ fields: labels, hull_fraction, family, L, B, D, nz_hull.
Optional: reference_labels, T, main_hull_fraction, sample_id, model_id, seed.
Native x,y,z, NEVER compact 64x32x32 encoding unless actually native.
"""
from __future__ import annotations
import argparse, csv, json, math, traceback
from pathlib import Path
import numpy as np
from .interface import EngineeringVolume, evaluate_volume, compare_pair


def _scalar(data, key, fallback=None):
    if key not in data.files:
        if fallback is not None:return fallback
        raise ValueError(f'NPZ missing required field {key}')
    x=np.asarray(data[key])
    if x.size!=1:raise ValueError(f'{key} expected scalar')
    return x.item()


def _finite_json(x):
    if isinstance(x,dict):return {k:_finite_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_finite_json(v) for v in x]
    if isinstance(x,(np.generic,)):return _finite_json(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x


def _flatten(x, prefix=''):
    out={}
    for k,v in x.items():
        name=prefix+k
        if isinstance(v,dict):out.update(_flatten(v,name+'_'))
        else:out[name]=v
    return out


def score_file(path:Path, *, equilibrium:bool=False, config_path=None) -> dict:
    with np.load(path,allow_pickle=False) as dat:
        frac=np.asarray(dat['hull_fraction'])
        kwargs=dict(family=str(_scalar(dat,'family')),L=float(_scalar(dat,'L')),
                    B=float(_scalar(dat,'B')),D=float(_scalar(dat,'D')),
                    nz_hull=int(_scalar(dat,'nz_hull')),
                    T=float(_scalar(dat,'T')) if 'T' in dat.files else None,
                    main_hull_fraction=dat['main_hull_fraction'] if 'main_hull_fraction' in dat.files else None,
                    sample_id=str(_scalar(dat,'sample_id',path.stem)),mask_provenance='npz:'+str(path))
        vol=EngineeringVolume(labels=dat['labels'],hull_fraction=frac,**kwargs)
        identity={'path':str(path),'sample_id':kwargs['sample_id'],
                  'model_id':str(_scalar(dat,'model_id','')),
                  'seed':str(_scalar(dat,'seed',''))}
        if 'reference_labels' in dat.files:
            ref=EngineeringVolume(labels=dat['reference_labels'],hull_fraction=frac,**kwargs)
            res=compare_pair(ref,vol,equilibrium=equilibrium,config_path=config_path)
        else:res=evaluate_volume(vol,equilibrium=equilibrium,config_path=config_path)
        return {**identity,'status':'ok',**_flatten(res)}


def main(argv=None):
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--input',type=Path,required=True,help='single .npz or directory of .npz')
    pa.add_argument('--output',type=Path,required=True,help='metrics.csv or .jsonl')
    pa.add_argument('--equilibrium',action='store_true',help='also solve trim and calibrated GM')
    pa.add_argument('--config',type=Path,help='override copied frozen P1A calibration')
    args=pa.parse_args(argv)
    paths=sorted(args.input.glob('*.npz')) if args.input.is_dir() else [args.input]
    if not paths:pa.error('no NPZ files found')
    rows=[]
    for p in paths:
        try:rows.append(score_file(p,equilibrium=args.equilibrium,config_path=args.config))
        except Exception as ex:rows.append({'path':str(p),'status':'error','error':f'{type(ex).__name__}: {ex}'})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.suffix.lower()=='.jsonl':
        with args.output.open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(_finite_json(row),allow_nan=False)+'\n')
    elif args.output.suffix.lower()=='.csv':
        names=sorted({k for r in rows for k in r})
        with args.output.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=names);w.writeheader()
            for row in rows:w.writerow(_finite_json(row))
    else:pa.error('--output extension must be .csv or .jsonl')
    failures=sum(r['status']!='ok' for r in rows)
    print(f'Wrote {len(rows)} records ({failures} input/evaluation errors) to {args.output}')
    if failures:print('Errors were retained with status=error, not silently removed.')
    return 1 if failures else 0

if __name__=='__main__':raise SystemExit(main())
