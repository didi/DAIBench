from functools import singledispatch

from collections import defaultdict, OrderedDict


def id(x):
    return x


def compose(f, g):
    return lambda x: f(g(x))


def register_dicts(fn, f):
    dicts = [defaultdict, OrderedDict, dict]
    for t in dicts:
        fn.register(t, f)


@singledispatch
def fmap(c, _):
    raise NotImplementedError(f"fmap not implemented for {c}")


@fmap.register(list)
def _l(c, fn):
    return [fn(v) for v in c]


@fmap.register(tuple)
def _t(c, fn):
    return (fn(v) for v in c)


@fmap.register(set)
def _s(c, fn):
    return {fn(v) for v in c}


@fmap.register(dict)
def _d(c, fn):
    return {k: fn(v) for k, v in c.items()}


@fmap.register(OrderedDict)
def _od(c, fn):
    return OrderedDict([(k, fn(v)) for k, v in c.items()])


@fmap.register(defaultdict)
def _dd(c, fn):
    r = c.default_factory()
    r.update({k: fn(v) for k, v in c.items()})
    return r


def nestfmap(xs, node_fn, leaf_fn):
    def recur(x):
        if x.__class__ in fmap.registry.keys():
            return fmap(node_fn(x), recur)
        else:
            return leaf_fn(x)

    return recur(xs)


@singledispatch
def keyvalmap(d, fn_k, fn_v):
    return d


@keyvalmap.register(dict)
def _keyvalmap_d(d, fn_k, fn_v):
    return {fn_k(k, v): fn_v(k, v) for k, v in d.items()}


@keyvalmap.register(OrderedDict)
def _keyvalmap_od(d, fn_k, fn_v):
    return OrderedDict([(fn_k(k, v), fn_v(k, v)) for k, v in d.items()])


@keyvalmap.register(defaultdict)
def _keyvalmap_dd(d, fn_k, fn_v):
    r = d.default_factory()
    r.update({fn_k(k, v): fn_v(k, v) for k, v in d.items()})
    return r


#register_dicts(keyvalmap, _keyvalmap_d)


def nestkeyvalmap(xs, fn_k, fn_v):
    return nestfmap(xs, lift(keyvalmap, fn_k, fn_v), id)


def lift(fn, *f):
    return lambda c: fn(c, *f)


@singledispatch
def yieldmap(c, fn):
    raise NotImplementedError(f"yieldmap not implemented for {c}")


def _ym_l(l, fn):
    for i in l:
        yield from fn(i)


yieldmap.register(list, _ym_l)
yieldmap.register(tuple, _ym_l)
yieldmap.register(set, _ym_l)


@yieldmap.register(dict)
def _ym_d(d, fn):
    for _, v in d.items():
        yield from fn(v)


register_dicts(yieldmap, _ym_d)


@singledispatch
def flatten_dict(v, prefix=[], sep="."):
    yield ((sep.join(prefix), v))


def _fd(d, prefix=[], sep="."):
    for k, v in d.items():
        yield from flatten_dict(v, prefix=prefix + [k], sep=sep)


register_dicts(flatten_dict, _fd)


@flatten_dict.register(list)
def _fl(l, prefix=[], sep="."):
    for i, v in enumerate(l):
        yield from flatten_dict(v, prefix=prefix + [str(i)], sep=sep)


def deep_put(d, klst, v):
    assert len(klst) > 0
    k, *tail = klst
    if len(tail) == 0:
        d[k] = v
    else:
        deep_put(d[k], tail, v)


InfDD = (lambda idd: idd(idd))(lambda f: lambda: defaultdict(f(f)))


def reconstruct(kv_list, sep="."):
    d = InfDD()

    for k, v in kv_list:
        keys = k.split(sep)
        deep_put(d, keys, v)

    return d


@singledispatch
def ttl(d):
    pass


def _ttl_d(d):
    islist = all(map(lambda x: x.isdigit(), d.keys()))
    if islist:
        return [d[str(i)] for i in sorted(map(int, d.keys()))]
    else:
        return d


register_dicts(ttl, _ttl_d)


def rebuild_lists(d):
    return nestfmap(d, ttl, id)
