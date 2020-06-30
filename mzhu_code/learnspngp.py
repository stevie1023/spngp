import numpy as np 
from collections import Counter
from scipy.stats import beta,iqr

class Color():
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    LIGHTBLUE = '\033[96m'
    FADE      = '\033[90m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def flt(flt):
        r = "%.4f" % flt
        return f"{Color.FADE}{r}{Color.ENDC}"
    
    @staticmethod
    def bold(txt):
        return f"{Color.OKGREEN}{txt}{Color.ENDC}"
    
    @staticmethod
    def val(flt, **kwargs):
        c = kwargs.get('color', 'yellow')
        e = kwargs.get('extra', '')
        f = kwargs.get('f', 4)

        if flt != float('-inf'):
            r = f"%.{f}f" % flt if flt != None else None
        else:
            r = '-∞'
            
        colors = {
            'yellow': Color.WARNING,
            'blue':  Color.OKBLUE,
            'orange': Color.FAIL,
            'green': Color.OKGREEN,
            'lightblue': Color.LIGHTBLUE
        }
        return f"{colors.get(c)}{e}{r}{Color.ENDC}"
        
class Mixture:
    def __init__(self, **kwargs):
        self.maxs      = kwargs['maxs']
        self.mins      = kwargs['mins']
        self.deltas    = dict.get(kwargs, 'deltas', [])
        self.spreads   = self.maxs - self.mins
        self.dimension = dict.get(kwargs, 'dimension', None)
        self.children  = dict.get(kwargs, 'children', [])
        self.depth     = dict.get(kwargs, 'depth', 0)
        self.n         = kwargs['n']
        self.parent    = dict.get(kwargs, 'parent', None)
        self.splits    = dict.get(kwargs, 'splits', [])
        self.idx = dict.get(kwargs, 'idx', [])
        #assert np.all(self.spreads > 0)

    def __repr__(self, level = 0):
        _dim = Color.val(self.dimension, f=0, color='orange', extra="dim=")
        _dep = Color.val(self.depth, f=0, color='yellow', extra="dep=")
        _nnn = Color.val(self.n, f=0, color='green', extra="n=")
        _rng = [f"{round(self.mins[i],2)} - {round(self.maxs[i],2)}" for i, _ in enumerate(self.mins)]
        _rng = ", ".join(_rng)
        
        if self.mins.shape[0] > 4:
            _rng = "..."

        _sel = " "*(level) + f"✚ Mixture [{_rng}] {_dim} {_dep} {_nnn}"

        if level <= 100:
            for split in self.children:
                _sel += f"\n{split.__repr__(level+2)}"
        else:
            _sel += " ..."
        return f"{_sel}"

class Separator:
    def __init__(self, **kwargs):
        self.split     = kwargs['split']
        self.dimension = kwargs['dimension']
        self.depth     = kwargs['depth']
        self.children  = kwargs['children']
        self.parent    = kwargs['parent']

    def __repr__(self, level=0):
        _sel = " "*(level) + f"ⓛ Separator dim={self.dimension} split={round(self.split, 2)}"
        
        for child in self.children:
            _sel += f"\n{child.__repr__(level+2)}"

        return _sel

class GPMixture:
    def __init__(self, **kwargs):
        self.mins      = kwargs['mins']
        self.maxs      = kwargs['maxs']
        self.idx       = dict.get(kwargs, 'idx', [])
        self.parent    = kwargs['parent']

    def __repr__(self, level = 0):
        _rng = [f"{round(self.mins[i],2)} - {round(self.maxs[i],2)}" for i, _ in enumerate(self.mins)]
        _rng = ", ".join(_rng)
        if self.mins.shape[0] > 4:
            _rng = "..."

        return " "*(level) + f"⚄ GPMixture [{_rng}] n={len(self.idx)}"

def _cached_gp(cache, **kwargs):
    min_, max_ = list(kwargs['mins']), list(kwargs['maxs'])
    cached = dict.get(cache, (*min_, *max_))
    cached1 = dict.get(cache, (*min_, *max_))
    cached2 = dict.get(cache, (*min_, *max_))
    if not cached:
        cache[(*min_,*max_)] = GPMixture(**kwargs), GPMixture(**kwargs),GPMixture(**kwargs)

        
    return cache[(*min_,*max_)]

def query(X, mins, maxs, skipleft=False):
    mask, D = np.full(len(X), True), X.shape[1]
    for d_ in range(D):
        if not skipleft:
            mask = mask & (X[:, d_] >= mins[d_]) & (X[:,d_] <= maxs[d_])
        else:
            mask = mask & (X[:, d_] > mins[d_]) & (X[:,d_] <= maxs[d_])

    return np.nonzero(mask)[0]


def get_splits(X, dd, **kwargs):
    meta      = dict.get(kwargs, 'meta', [""] * X.shape[1])
    max_depth = dict.get(kwargs, 'max_depth', 8)
    log       = dict.get(kwargs, 'log', False)
    
    features_mask = np.zeros(X.shape[1])
    splits        = np.zeros((X.shape[1], dd-1))
    quantiles     = np.quantile(X, np.arange(0, 1, 1/dd)[1:], axis=0).T
    for i, var in enumerate(quantiles):
        include = False
        if dd == 2:
            spread = np.sum(X[:, i] < var[0]) - np.sum(X[:, i] >= var[0])
            
            if np.abs(spread) < X.shape[0]/12:
                include = True
        elif len(np.unique(np.round(var, 8))) == len(var):
                include = True
        
        if include:
            features_mask[i] = 1
            splits[i]        = np.array(var)

            if np.sum(features_mask) <= max_depth and meta and log: 
                print(i, "\t", meta[i], var)
            else: pass #print('.', end = '')

    return splits, features_mask

def build_bins(**kwargs):
    X                   = kwargs['X']
    max_depth           = dict.get(kwargs, 'max_depth',               8)
    min_samples         = dict.get(kwargs, 'min_samples',             0)
    max_samples         = dict.get(kwargs, 'max_samples',         10**4)
    qd                  = dict.get(kwargs, 'qd',                      0)
    log                 = dict.get(kwargs, 'log',                 False)
    jump                = dict.get(kwargs, 'jump',                False)
    reduce_branching    = dict.get(kwargs, 'reduce_branching',    False)
    randomize_branching = dict.get(kwargs, 'randomize_branching', False)

    # splits, features_mask = get_splits(X, qd, meta=dict.get(kwargs, 'meta', None), log=log)
    
    root_mixture_opts = {
        'mins':      np.min(X, 0),  #min & max of every feature
        'maxs':      np.max(X, 0), 
        'n':         len(X), #here the valid length of the data(for every feature)
        'parent':    None,
        'dimension': np.argmax(np.var(X, axis=0)),
        'idx': X
        #the index of the features
    }

    nsplits            = Counter()
    root_node          = Mixture(**root_mixture_opts)
    to_process, cache  = [root_node], dict()

    while len(to_process):
        node = to_process.pop()
        
        if type(node) is not Mixture:
            continue

        X = node.idx
        d_selected = np.argsort(-np.var(X, axis=0))
        d = node.dimension
        mins, maxs = np.min(X, 0), np.max(X, 0)
        splits, features_mask = get_splits(X, qd, meta=dict.get(kwargs, 'meta', None), log=log)
        d2 = d_selected[1]
        d3 = d_selected[2]

        fit_lhs = node.mins < splits[:,  0]# the node is the Mixture now(include all data for every feature
        fit_rhs = node.maxs > splits[:, -1]
        create  = np.logical_and(fit_lhs, fit_rhs)
        create  = np.logical_and(create, features_mask)
    
        # Preprocess splits
        node_splits = []#store only splits in one dimension(feature) in every loop, d is changing in different loops

        node_splits2 = []
        node_splits3 = []

        for node_split in splits[d]:
            # We skip the split completely if it is outside of 
            # the scope of the data in this dimension. parent split 
            # has the data already. this saves us form n = 0 mixtures
            if node_split <= node.mins[d] or node_split >= node.maxs[d]:
                continue

            node_splits.append(node_split)    #store the valid splits

        for node_split in splits[d2]:
            if node_split <= node.mins[d2] or node_split >= node.maxs[d2]:
                continue

            node_splits2.append(node_split)

        for node_split in splits[d3]:
            if node_split <= node.mins[d3] or node_split >= node.maxs[d3]:
                continue

            node_splits3.append(node_split)
        method1=[]
        method2=[]
        method3=[]
        method4=[]
        method5=[]
        method6=[]
        method7=[]
        method8=[]
        a = iqr(X, axis=0)
        r = beta.rvs(2, 2, size = 4)
        v1 = maxs[d] - mins[d]
        v2 = maxs[d2] - mins[d2]
        v3 = maxs[d3] - mins[d3]
        split11 = mins[d]+a[d]
        split12= mins[d2]+a[d2]
        split13 = mins[d3]+a[d3]


        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        sm = a[d] / (q1[d] + q3[d])
        sm2 = a[d2] / (q1[d2] + q3[d2])
        sm3 = a[d3] / (q1[d3] + q3[d3])
        split21 = (q1[d]+q3[d])/2
        split22 = (q1[d2]+q3[d2])/2
        split23 = (q1[d3]+q3[d3])/2
        split31 = 0.8 * (v1 * r[d] + mins[d]) + 0.2 * split11
        split32 = 0.8 * (v2 * r[d2] + mins[d2]) + 0.2 * split12
        split33 = 0.8 * (v3 * r[d3] + mins[d3]) + 0.2 * split13

        split41 = 0.8 * (v1 * r[d] + mins[d]) + 0.2 * split21
        split42 = 0.8 * (v2 * r[d2] + mins[d2]) + 0.2 * split22
        split43 = 0.8 * (v3 * r[d3] + mins[d3]) + 0.2 * split23

        split51 = 0.2 * (v1 * r[d] + mins[d]) + 0.8 * split11
        split52 = 0.2 * (v2 * r[d2] + mins[d2]) + 0.8 * split12
        split53 = 0.2 * (v3 * r[d3] + mins[d3]) + 0.8 * split13

        split61 = 0.2 * (v1 * r[d] + mins[d]) + 0.8 * split21
        split62 = 0.2 * (v2 * r[d2] + mins[d2]) + 0.8 * split22
        split63 = 0.2 * (v3 * r[d3] + mins[d3]) + 0.8 * split23

        split71 = 0.1 * (v1 * r[d] + mins[d]) + 0.9 * split11
        split72 = 0.1 * (v2 * r[d2] + mins[d2]) + 0.9 * split12
        split73 = 0.1 * (v3 * r[d3] + mins[d3]) + 0.9 * split13

        split81 = 0.5 * (v1 * sm + mins[d]) + 0.5 * split11
        split82 = 0.5 * (v2 * sm2 + mins[d2]) + 0.5 * split12
        split83 = 0.5 * (v3 * sm3 + mins[d3]) + 0.5 * split13
        # node_splits_all = [np.mean(node_splits), np.mean(node_splits2), np.mean(node_splits3)]
        # print('median:',node_splits_all_print)
        node_splits_all = [split61,split62,split63]

        # print('median:', [np.median(node_splits), np.median(node_splits2), np.median(node_splits3)])
        # print('method1:',[split11,split12,split13])
        # print('method2:',[split21,split22,split23])
        # print('method3:',[split31,split32,split33])
        # print('method4:',[split41,split42,split43])
        # print('method5:',[split51, split52, split53])
        # print('method6:',[split61, split62, split63])
        # print('method7:',[split71, split72, split73])
        # print('method8:',[split81, split82, split83])


        if reduce_branching and node.depth >= 1:

            # node_splits_all = [np.mean(node_splits)]
            node_splits_all = [split61,split62]
            # node_splits_all = [np.median(X,axis=0)[d],np.median(X,axis=0)[d2]]

        if len(node_splits_all) == 0: raise Exception('1')

        d = [d, d2, d3]
        i = 0
        j = 0
        for split in node_splits_all:# again operate in splits in one dimension

            create_left  = create.copy()
            create_right = create.copy()
            create_left[d[i]]  = split != node_splits[0]
            create_right[d[i]] = split != node_splits[-1] #create left(right)nodes for splits other than the first/last split
            # no left nodes for the first split and no right nodes for the last split

            if jump:
                # We force a new dimension for every child 
                # on the same split level
                create_left[d[i]], create_right[d[i]]      = False, False
                create_right[np.argmax(create_left)] = False
            else:
                # We dont create new mixture in the limits
                create_left[d[i]]  = split != node_splits_all[0]
                create_right[d[i]] = split != node_splits_all[-1]
    
            new_maxs, new_mins       = node.maxs.copy(), node.mins.copy()
            new_maxs[d[i]], new_mins[d[i]] = split, split

            
            idx_left   = query(X, node.mins, new_maxs,  skipleft=False)
            idx_right  = query(X, new_mins,  node.maxs, skipleft=True)#return the first index of data that satisfies certain requirements
            print('left_1:', len(idx_left))
            print('right_1:', len(idx_right))

            next_depth = node.depth+1

            loop = [
                ('left',  create_left,  idx_left,  node.mins, new_maxs),
                ('right', create_right, idx_right, new_mins, node.maxs)
            ]
            
            results = []
            k=0
            for _, create_mixture, idx, mins, maxs, in loop:
                if min_samples == 0:
                    min_samples = min(len(idx_left), len(idx_right)) + 1

                can_create     = np.any(create_mixture)
                big_enough     = len(idx)   >= min_samples
                not_too_big    = len(idx)   <= max_samples
                not_too_deep   = next_depth  < max_depth

                if randomize_branching:
                    next_dimension = np.random.choice(np.where(create_mixture)[0])
                else:
                    # next_dimension = np.argmax(create_mixture) # the index of the newly spliting place is the next_dimension
                    x_idx = X[idx]
                    next_dimension = np.argmax(np.var(x_idx, axis=0))
                mixture_opts = {
                        'mins':      mins,
                        'maxs':      maxs,
                        'depth':     next_depth,
                        'dimension': next_dimension,
                        'n':         len(idx),
                        'idx':       x_idx
                    #the number of left/right new splits
                }                                   #newly genenrated mixture nodes for the next dimension
                k+=1



                if all([can_create, big_enough, not_too_deep, not_too_big]):
                    results.append(Mixture(**mixture_opts))

                elif can_create and len(idx) > max_samples:
                    print("Forcing a Mixture...")
                    results.append(Mixture(**mixture_opts))
                elif len(idx):
                    if len(idx) > max_samples:
                        print(f"Had to create a GP with n={len(idx)} because we ran out of splits.")
                    gp = _cached_gp(cache, mins=mins, maxs=maxs, idx=idx, parent=None)
                    gp1 = _cached_gp(cache, mins=mins, maxs=maxs, idx=idx, parent=None)#modified
                    gp2 = _cached_gp(cache, mins=mins, maxs=maxs, idx=idx, parent=None)
                    results.append((gp))

            j+=1


            if len(results) == 2:
                to_process.extend(results) #results are put into the root_reigon
                separator_opts = {
                    'depth': node.depth,
                    'dimension':      d[i],
                    'split':      split,
                    'children': results, 
                    'parent':       None
                }
                node.children.append(Separator(**separator_opts))  #create product nodes for every mixture node

            elif len(results) == 1:
                node.children.extend(results)
                to_process.extend(results)
            else:
                raise Exception('1')
            i += 1

    # print('method1:',method1)
    #
    # print('method2:', method2)
    #
    # print('method2:', method3)
    #
    # print('method3:', method4)
    #
    # print('method4:', method5)
    #
    # print('method5:', method6)
    #
    # print('method6:', method7)
    #
    # print('method7:', method8)
    gps = list(cache.values())
    # aaa = [len(gp.idx) for gp in gps]
    aaa = [len((list(gp))[0].idx) for gp in gps]
    c = (np.mean(aaa)**3)*len(aaa)
    r = 1-(c/(len(X)**3))
    #
    gpss=[]
    for gp in gps:
        gp=list(gp)
        gpss.extend(gp)
    print("Full:\t\t", len(X)**3, "\nOptimized:\t", int(c), f"\n#GP's:\t\t {len(gps)} ({np.min(aaa)}-{np.max(aaa)})", "\nReduction:\t", f"{round(100*r, 4)}%")
    print(f"nsplits:\t {nsplits}")
    print(f"Lengths:\t {aaa}\nSum:\t\t {sum(aaa)} (N={len(X)})")
    #
    return root_node, gpss