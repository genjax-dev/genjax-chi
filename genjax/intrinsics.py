import jax.core as core
from jax._src import abstract_arrays

# Trace primitive.
trace_p = core.Primitive("trace")


def _trace(addr, prim, *args):
    try:
        return trace_p.bind(*args, addr=addr, prim=prim)
    except:
        splice_p.bind(addr=addr)
        ret = prim(*args)
        unsplice_p.bind(addr=addr)
        return ret


def trace(addr, prim):
    return lambda *args: _trace(addr, prim, *args)


def trace_abstract_eval(*args, **kwargs):
    prim = kwargs["prim"]
    prim = prim()
    return prim.abstract_eval(*args)


trace_p.multiple_results = True
trace_p.def_abstract_eval(trace_abstract_eval)
trace_p.must_handle = True


# Hierarchical (call) addressing primitive.
splice_p = core.Primitive("splice")


def splice_abstract_eval(addr):
    return abstract_arrays.ShapedArray(shape=(0,), dtype=bool)


splice_p.def_abstract_eval(splice_abstract_eval)
splice_p.must_handle = True


# Hierarchical (call) addressing primitive.
unsplice_p = core.Primitive("unsplice")


def unsplice_abstract_eval(addr):
    return abstract_arrays.ShapedArray(shape=(0,), dtype=bool)


unsplice_p.def_abstract_eval(unsplice_abstract_eval)
unsplice_p.must_handle = True
