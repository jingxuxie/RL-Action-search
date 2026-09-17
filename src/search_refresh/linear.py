"""Small fitted linear critic experiment; no neural/GPU dependency.

Each time layer has its own linear regressor. Features share state and action
main effects plus two random interaction features, preventing cross-layer
leakage and leaving nontrivial within-layer misspecification.
"""
import numpy as np

class LayerwiseRidge:
    def __init__(self,model,counts,seed=0,ridge=1e-3):
        self.model=model; self.groups=[]
        rng=np.random.default_rng(seed)
        for h in range(model.H):
            states=np.flatnonzero(model.level==h); w=len(states); a=model.A
            one_s=np.repeat(np.eye(w),a,axis=0)
            one_a=np.tile(np.eye(a),(w,1))
            interact=rng.normal(size=(w*a,2))
            x=np.c_[np.ones(w*a),one_s,one_a,interact]
            weights=np.maximum(counts[states].ravel(),1).astype(float)
            weights/=weights.mean()
            fit=np.linalg.solve(x.T@(weights[:,None]*x)+ridge*np.eye(x.shape[1]),x.T*weights)
            self.groups.append((states,x,fit))

    def fit_predict(self,targets):
        out=np.empty_like(targets)
        for states,x,fit in self.groups:
            out[states]=(x@(fit@targets[states].ravel())).reshape(len(states),self.model.A)
        bounds=(self.model.H-self.model.level)[:,None]
        return np.clip(out,0,bounds)
