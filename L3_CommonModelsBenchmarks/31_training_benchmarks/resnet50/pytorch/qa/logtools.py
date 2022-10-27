import json
import math
from functools import singledispatch
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
import functools as ft

import func_helpers as fh

# DATACLASSES {{{


class DLLLLog:
    def __init__(self, log):
        self.log = log


class NVDFLog:
    def __init__(self, created, env, metadata, parameters, log):
        self.created = created
        self.env = env
        self.metadata = metadata
        self.parameters = parameters
        self.log = log

    def to_dict(self):
        return dict(
            created=self.created,
            env=self.env,
            metadata=self.metadata,
            parameters=self.parameters,
            log=self.log,
        )

    def __getitem__(self, key):
        return self.__dict__[key]


@fh.fmap.register(NVDFLog)
def _map_nvdflog(c, fn):
    return NVDFLog(
        fn(c.created), fn(c.env), fn(c.metadata), fn(c.parameters), fn(c.log)
    )


class NVDFLogTyped:
    def __init__(self, ts_created, nested_config, nested_log):
        self.ts_created = ts_created
        self.nested_config = nested_config
        self.nested_log = nested_log

    def to_dict(self):
        return dict(
            ts_created=self.ts_created,
            nested_config=self.nested_config,
            nested_log=self.nested_log,
        )

    def __getitem__(self, key):
        return self.__dict__[key]


@fh.fmap.register(NVDFLogTyped)
def _map_nvdflogtyped(c, fn):
    return NVDFLogTyped(
        fn(c.ts_created),
        fn(c.obj_env),
        fn(c.obj_metadata),
        fn(c.obj_parameters),
        fn(c.nested_log),
    )


# }}}

# ADD STUFF {{{


def add_metadata(nvdflog, metadata):
    metadata = sanitize(metadata)
    comb_meta = combine_nested_dicts(nvdflog.metadata, metadata)

    return NVDFLog(
        created=nvdflog.created,
        env=nvdflog.env,
        metadata=comb_meta,
        parameters=nvdflog.parameters,
        log=nvdflog.log,
    )


def add_env(nvdflog, env):
    env = sanitize(env)
    comb_env = dict(*nvdflog.env)
    comb_env.update(env)

    return NVDFLog(
        created=nvdflog.created,
        env=comb_env,
        metadata=nvdflog.metadata,
        parameters=nvdflog.parameters,
        log=nvdflog.log,
    )


# }}}

# TYPE HELPERS {{{

def strip_type(s):
    return "_".join(s.split("_")[1:])


kibana_full_map = {
    bool: "b_",
    int: "l_",
    float: "d_",
    str: "s_",
    dict: "obj_",
    type(None): "ns_",
}

kibana_ni_map = {bool: "ni_", int: "ni_", float: "ni_", str: "ni_", dict: "obj_"}


def type_key(map, key, type):
    #    if key == "timestamp":
    #        return "ts_timestamp"

    return map.get(type, "ns_") + key


def type_val(key, val, type):
    if key == "timestamp":
        return int(float(val) * 1000)

    if type is None:
        return None

    if isinstance(val, list) or isinstance(val, tuple):
        return fh.fmap(val, type)

    return type(val)


def merge_types(t1, t2):
    value_types = [bool, int, float, str]
    compound_types = {tuple, dict, list}

    if t1 in compound_types and t2 in compound_types:
        assert t1 == t2
        return t1

    if t1 in value_types and t2 in value_types:
        return t1 if value_types.index(t1) > value_types.index(t2) else t2

    return None


def key_type_mapping(ktg):
    mapping = {}
    for k, t in ktg:
        if k not in mapping:
            mapping[k] = merge_types(t, t)
        else:
            mapping[k] = merge_types(mapping[k], t)
    return mapping


@singledispatch
def key_type_gen(c):
    if c.__class__ in fh.yieldmap.registry.keys():
        yield from fh.yieldmap(c, key_type_gen)


def _ktg_d(d):
    for k, v in d.items():
        yield k, None if v is None else value_type(v)
        yield from key_type_gen(v)


fh.register_dicts(key_type_gen, _ktg_d)


def value_type(f):
    if isinstance(f, list) or isinstance(f, tuple):
        if len(f) == 0:
            return None
        return ft.reduce(merge_types, map(type, f))
    else:
        return type(f)


# }}}

# LOG CONVERSION {{{


def dll2nvdf(dll_log):
    assert isinstance(dll_log, DLLLLog)
    created = datetime.now().timestamp()
    env = {}
    split_by_type = div_by_key(dll_log.log, key="type")

    metadata = transpose_dict(
        split_by_type["METADATA"],
        get_key=lambda d: d["metric"],
        get_data=lambda d: dict(key_name=d["metric"], **d["metadata"]),
    )

    split_by_step = div_by_key(split_by_type["LOG"], key="step", eq_class_fn=type)

    parameters = div_by_key(split_by_step[str], key="step")
    if "PARAMETER" not in parameters.keys():
        print("!!!!!!!! NVDF CONVERTER WARNING PARAMETERS NOT LOGGED !!!!!!!!")
    if len(parameters.keys()) > 1:
        print("!!!!!!!! NVDF CONVERTER WARNING LOG IS NOT CONVENTION COMPLIANT, STEP != PARAMETERS or tuple !!!!!!!!")

    parameters = transpose_dict(
        parameters["PARAMETER"],
        get_key=lambda d: d["step"],
        get_data=lambda d: d["data"],
    )["PARAMETER"]

    log = transpose_dict(
        split_by_step[list]
        + split_by_step[tuple]
        + split_by_step[float]
        + split_by_step[int],
        get_key=lambda d: tuple(d["step"])
        if isinstance(d["step"], Iterable)
        else (d["step"],),
        get_data=lambda d: dict(
            timestamp=d["timestamp"], elapsed=d.get("elapsedtime", 0), **d["data"]
        ),
    )

    step_length = max(1, max(map(len, log.keys())))

    pad_tuple = lambda t, _: (t + tuple([-1] * step_length))[:step_length]

    log = fh.keyvalmap(log, pad_tuple, lambda k, v: v)

    log = [
        {**{f"step_{i}": it for i, it in enumerate(step)}, **v}
        for step, v in log.items()
    ]

    return sanitize(NVDFLog(datetime.now().timestamp(), env, metadata, parameters, log))


def sanitize(x):
    return fh.nestkeyvalmap(x, lambda k, v: sanitize_str(k), lambda k, v: v)


def type_log_deprec(nvdflog):
    assert isinstance(nvdflog, NVDFLog)
    nvdflog = sanitize(nvdflog)
    type_maps = fh.fmap(nvdflog, fh.compose(key_type_mapping, key_type_gen))
    type_field = lambda map: lambda f, t: fh.nestkeyvalmap(
        f, lambda k, v: type_key(map, k, t[k]), lambda k, v: type_val(k, v, t[k])
    )
    type_field_full = type_field(kibana_full_map)
    type_field_ni = type_field(kibana_ni_map)
    created = type_val(
        "timestamp", nvdflog.created, id
    )  # ID is not used in this case, so this could be anything
    log = type_field_ni(nvdflog.log, type_maps.log)
    env = type_field_full(nvdflog.env, type_maps.env)
    parameters = type_field_ni(nvdflog.parameters, type_maps.parameters)
    metadata = type_field_ni(nvdflog.metadata, type_maps.metadata)
    metadata = {
        m: {
            type_key(kibana_ni_map, "kibana_prefix", str): type_key(
                kibana_ni_map, "", type_maps.log[strip_type(m)]
            ),
            **md,
        }
        if strip_type(m) in type_maps.log.keys()
        else md
        for m, md in metadata.items()
    }

    return NVDFLogTyped(created, env, metadata, parameters, log)


py_types = [
    (float, "float"),
    (int, "int"),
    (str, "str"),
    (bool, "bool"),
    (type(None), "None"),
]

py_type_to_str = {t: s for t, s in py_types}

str_to_py_type = {s: t for t, s in py_types}
str_to_py_type["None"] = lambda _: None


def kibana_entry(t):
    isnotfinite = isinstance(t[1], float) and not math.isfinite(t[1])
    kibana_type = "s_" if isnotfinite else kibana_full_map[type(t[1])]
    kibana_value = str(t[1]) if isnotfinite else t[1]

    return {
        "s_field": t[0],
        kibana_type + "value": kibana_value,
        "s_type": py_type_to_str[type(t[1])],
    }


def from_kibana_entry(d):
    value_key = list(filter(lambda k: k.endswith("_value"), d.keys()))
    assert len(value_key) == 1
    value = str_to_py_type[d["s_type"]](d[value_key[0]])

    return d["s_field"], str_to_py_type[d["s_type"]](d[value_key[0]])


def type_log(nvdflog):
    nvdflog = sanitize(nvdflog)
    # assert isinstance(nvdflog, NVDFLog)
    log = {
        "ts_created": type_val("timestamp", nvdflog.created, id),
        "nested_config": list(
            map(
                kibana_entry,
                fh.flatten_dict(
                    {
                        "env": nvdflog.env,
                        "metadata": nvdflog.metadata,
                        "parameters": nvdflog.parameters,
                    }
                ),
            )
        ),
        "nested_log": [
            {"nested_entry": list(map(kibana_entry, fh.flatten_dict(entry)))}
            for entry in nvdflog.log
        ],
    }

    return NVDFLogTyped(**log)


def untype_log(nvdflogtyped):
    config = fh.rebuild_lists(
        fh.reconstruct(map(from_kibana_entry, nvdflogtyped["nested_config"]))
    )
    log = [
        fh.rebuild_lists(fh.reconstruct(map(from_kibana_entry, obj["nested_entry"])))
        for obj in nvdflogtyped["nested_log"]
    ]
    return NVDFLog(
        created=nvdflogtyped.ts_created / 1000,
        log=log,
        env=config["env"],
        metadata=config["metadata"],
        parameters=config["parameters"],
    )


def untype_log_deprec(nvdflogtyped):
    assert isinstance(nvdflogtyped, NVDFLogTyped)
    stripped = fh.nestkeyvalmap(
        nvdflogtyped, lambda k, v: strip_type(k), lambda k, v: v
    )
    return NVDFLog(
        created=stripped.ts_created / 1000,
        env=stripped.obj_env,
        metadata=stripped.obj_metadata,
        parameters=stripped.obj_parameters,
        log=stripped.nested_log,
    )


# }}}

# MISC {{{


def sanitize_str(string):
    # illegal_chars = "/\!@#$%^&*.{}[]()"
    illegal_chars = "."

    return "".join(map(lambda x: "_" if x in illegal_chars else x, string))


def load_dlll_log(file, prefix="DLLL "):
    data = []
    for line in file.readlines():
        if line.startswith(prefix):
            data.append(json.loads(line[len(prefix) :]))

    return DLLLLog(data)


def load_nvdflog(file):
    log = json.load(file)

    return NVDFLog(**log)


def load_nvdflog_typed(file):
    log = json.load(file)

    return NVDFLogTyped(**log)


# }}}

# DICT MANIPULATION {{{


def combine_nested_dicts(d1, d2):
    cd = defaultdict(dict)
    for key in list(d1.keys()) + list(d2.keys()):
        cd[key].update(d1.get(key, {}))
        cd[key].update(d2.get(key, {}))

    return cd


def div_by_key(data, key, eq_class_fn=lambda x: x):
    div_data = defaultdict(list)
    for d in data:
        div_data[eq_class_fn(d[key])].append(d)

    return div_data


def transpose_dict(data, get_key, get_data):
    transposed_data = defaultdict(dict)
    for d in data:
        transposed_data[get_key(d)].update(get_data(d))

    return transposed_data


def metadata_from_baseline(baseline):
    return {m: {"baseline": b} for m, b in baseline.items()}


# }}}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--env", default=None)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dl_log = load_dlll_log(open(args.log, "r"))
    env = {} if args.env is None else json.load(open(args.env, "r"))
    meta = {} if args.meta is None else json.load(open(args.meta, "r"))
    baseline = {} if args.baseline is None else json.load(open(args.baseline, "r"))

    nv_log = dll2nvdf(dl_log)
    nv_log = add_metadata(nv_log, meta)
    nv_log = add_env(nv_log, env)
    snv_log = sanitize(nv_log)
    snv_log = add_metadata(snv_log, metadata_from_baseline(baseline))
    nvdflog = type_log(snv_log)

    if args.output is None:
        print(json.dumps(nvdflog.to_dict(), indent=4))
    else:
        with open(args.output, 'w') as f:
            f.write(json.dumps(nvdflog.to_dict(), indent=4))
