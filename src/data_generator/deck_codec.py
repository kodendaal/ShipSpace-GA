"""Diagnostic one-function-per-(x,y,deck) codec, using a supplied geometry map.

deck_index[x,y,z] is geometry metadata, with -1 outside the supplied envelope.
Its construction must not depend on unknown functional labels at inference.
Each compact cell represents all native occupied cells mapped to it. The
encoder chooses the most frequent function (lowest class ID breaks ties).
This projection is generally lossy; inspect mixed columns and reconstruction.
"""
import numpy as np


def build_geometry_map(hull_fraction, full_envelope, hull_bands, ss_layers_per_deck, *, mode='logical', max_slots=32):
    """Build the decoding coordinates from supplied geometry, never functions.

    hull_bands holds native [z0,z1,...] intervals from the PRE-assignment deck
    recipe. Full envelope is an explicit design condition. The cache supplies
    the local hull top, so settled SS decks need not have one global elevation.
    mode='logical': one slot per hull band and per logical SS deck.
    mode='compact32': start with all hull slices, reserving one SS slot per deck;
    merge within hull bands only if the slot budget requires it. The greedy
    merge criterion depends only on fractional hull geometry.
    mode='compact32_protected': permit those merges only in UPPER-role bands
    (role 2). This protects DB/lower hull detail for layouts from this
    generator; it is not a guarantee for other generators or real GAs.
    """
    hull=np.asarray(hull_fraction,dtype=np.float32)
    envelope=np.asarray(full_envelope,dtype=bool)
    if hull.ndim!=3 or envelope.ndim!=3 or hull.shape[:2]!=envelope.shape[:2] or hull.shape[2]>envelope.shape[2]:
        raise ValueError('Invalid hull/envelope dimensions')
    plans=[(int(a),int(b)) for a,b,*_ in hull_bands]
    if not plans or plans[0][0]!=0 or plans[-1][1]!=hull.shape[2] or any(a>=b for a,b in plans) or any(b!=c for (a,b),(c,d) in zip(plans[:-1],plans[1:])):
        raise ValueError('Hull bands must form a contiguous ordered partition')
    layers=np.asarray(ss_layers_per_deck,dtype=np.int64)
    if layers.ndim!=1 or np.any(layers<1):raise ValueError('Invalid SS deck thicknesses')
    if mode=='logical':
        groups=[(a,b,k) for k,(a,b) in enumerate(plans)]
    elif mode in ('compact32','compact32_protected'):
        roles=[int(row[-1]) for row in hull_bands]
        if mode=='compact32_protected' and any(len(row)<3 for row in hull_bands):
            raise ValueError('Protected compression requires pre-assignment hull-band roles')
        groups=[(z,z+1,next(k for k,(a,b) in enumerate(plans) if a<=z<b)) for z in range(hull.shape[2])]
        budget=int(max_slots)-len(layers)
        if budget<len(plans):raise ValueError('Slot budget cannot preserve all hull bands')
        while len(groups)>budget:
            choices=[]
            for q,((a,b,k),(c,d,l)) in enumerate(zip(groups[:-1],groups[1:])):
                if k==l and (mode!='compact32_protected' or roles[k]==2):
                    cost=float(np.abs(hull[:,:,a:b].mean(axis=2)-hull[:,:,c:d].mean(axis=2)).sum())
                    choices.append((cost,q))
            if not choices:raise ValueError('No eligible within-band merge')
            _,q=min(choices);a,b,k=groups[q];c,d,l=groups[q+1]
            groups[q:q+2]=[(a,d,k)]
    else:raise ValueError('Unknown coordinate-map mode')
    occ=hull>.10
    top=np.where(occ.any(axis=2),hull.shape[2]-1-np.argmax(occ[:,:,::-1],axis=2),-1)
    native_z=np.arange(envelope.shape[2])[None,None,:]
    ss=envelope & (native_z>top[:,:,None])
    if np.any(ss & (top[:,:,None]<0)):raise ValueError('SS column has no supporting hull')
    result=np.full(envelope.shape,-1,np.int16)
    for k,(a,b,_) in enumerate(groups):result[:,:,a:b]=k
    x,y,z=np.nonzero(ss);level=z-top[x,y]-1
    logical=np.searchsorted(np.cumsum(layers),level,side='right')
    if np.any(logical>=len(layers)):raise ValueError('Envelope exceeds declared SS stack')
    result[x,y,z]=len(groups)+logical
    result[~envelope]=-1
    if np.any(result[envelope]<0):raise ValueError('Envelope has unmapped cells')
    return {'deck_index':result,'slot_count':len(groups)+len(layers),'hull_groups':groups}


def encode_decks(labels, deck_index, n_decks, n_classes=11):
    lab = np.asarray(labels)
    deck = np.asarray(deck_index)
    if lab.shape != deck.shape or lab.ndim != 3:
        raise ValueError('Expected matching 3D label and deck-index arrays')
    inside = deck >= 0
    if inside.any() and (deck[inside].max() >= n_decks or lab[inside].min() < 0 or lab[inside].max() >= n_classes):
        raise ValueError('Deck or function index out of range')
    nx, ny, _ = lab.shape
    xy = np.broadcast_to(np.arange(nx*ny).reshape(nx,ny,1),lab.shape)
    keys = xy[inside]*n_decks + deck[inside]
    count = np.bincount(keys*n_classes+lab[inside],minlength=nx*ny*n_decks*n_classes)
    count = count.reshape(nx,ny,n_decks,n_classes)
    occupied_count = count.sum(axis=-1)
    compact = count.argmax(axis=-1).astype(np.uint8)
    compact[occupied_count == 0] = 10
    return {'labels':compact,'occupied_cell_count':occupied_count,
            'mixed_function_columns':(count>0).sum(axis=-1)>1,
            'minimum_native_cell_disagreements':int(occupied_count.sum()-count.max(axis=-1).sum())}


def decode_decks(compact_labels, deck_index, empty_class=10):
    compact = np.asarray(compact_labels)
    deck = np.asarray(deck_index)
    if compact.ndim!=3 or deck.ndim!=3 or compact.shape[:2]!=deck.shape[:2]:
        raise ValueError('Mismatched XY dimensions')
    inside=deck>=0
    if inside.any() and deck[inside].max()>=compact.shape[2]:
        raise ValueError('Deck index exceeds compact tensor')
    result=np.full(deck.shape,empty_class,dtype=compact.dtype)
    x,y,z=np.nonzero(inside)
    result[x,y,z]=compact[x,y,deck[x,y,z]]
    return result


def reduce_geometry(deck_index, occupied_fraction, spacing_m, n_decks):
    """Volume and first moments for decoded piecewise-constant predictions.

These quantities follow geometry only. They do not make an unrepresentable
mixed column lossless. They reproduce physical sums for the DECODED labels.
"""
    deck=np.asarray(deck_index);alpha=np.asarray(occupied_fraction,dtype=np.float64)
    if deck.shape!=alpha.shape or deck.ndim!=3:
        raise ValueError('Geometry arrays must match')
    inside=deck>=0;nx,ny,nz=deck.shape
    dx,dy,dz=map(float,spacing_m)
    x,y,z=np.nonzero(inside)
    keys=(x*ny+y)*n_decks+deck[inside]
    volume=alpha[inside]*dx*dy*dz
    shape=(nx,ny,n_decks);size=nx*ny*n_decks
    return {name:np.bincount(keys,weights=w,minlength=size).reshape(shape)
            for name,w in [('volume_m3',volume),('x_first_moment_m4',volume*(x+.5)*dx),
                           ('z_first_moment_m4',volume*(z+.5)*dz)]}
