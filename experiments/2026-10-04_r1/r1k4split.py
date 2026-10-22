"""ARC's factored k=4 nonlinear step with the kappa4 TREE births split into separate column blocks by graph type
(path, star, 3+2) instead of one summed block. Built from the reference's own source by three exact text edits
(each B accumulation closes into its own add_factors_ before the next graph type starts), so the arithmetic is the
reference's term for term; only the column layout changes: per ReLU [path | star | 3+2 | correction], 4n columns.
use: import r1k4split; r1k4split.install()  -> mlp_kprop.kprop_harmonic's factored k=4 step uses the split version.
"""
import inspect
import textwrap

import mlp_kprop.factor_k4 as fk4

_NEW = "B = torch.zeros((n, n, n), device=mean.device, dtype=mean.dtype)\n            B_jkl = B.permute(2, 0, 1)\n"


def _build():
    src = textwrap.dedent(inspect.getsource(fk4.factored_nonlin_kprop_k4))
    edits = [("            # Star with edges A(i, j), X(j, k), Y(j, l)\n",
              "            pK_1111.add_factors_((A, B))   # path block\n            " + _NEW
              + "            # Star with edges A(i, j), X(j, k), Y(j, l)\n"),
             ("            # 3+2 with edges A(i, j), B(j, k, l)\n",
              "            pK_1111.add_factors_((A, B))   # star block\n            " + _NEW
              + "            # 3+2 with edges A(i, j), B(j, k, l)\n")]
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = dict(vars(fk4))
    exec(compile(src, "<r1k4split>", "exec"), ns)
    return ns["factored_nonlin_kprop_k4"]


split_step = _build()


def install():
    fk4.factored_nonlin_kprop_k4 = split_step
    BLOCKS = 4
    return BLOCKS
