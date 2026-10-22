"""Task 4 of the port battery: the greedy stack on k3m + matched cut, re-measured on 10 MLPs at every step.

Start: the matched dead cut (deadcut = colcut = t, t = -2.25 by default; `--t -2.5` repeats the final stack at -2.5).
At every step each remaining survivor is added to the current stack and measured; the one with the best projected
adjusted (5.3906e-9 x raw / 2.3462e-8 x util / 24.835%, util from cost_pb.py) is kept; the loop stops when no
survivor improves the projected score. A candidate = (chain kwargs, cost_pb kwargs).

    PB_THREADS=1 ../../.venv-torch/bin/python t4_stack.py <step-json> <candidate names...>     (one step, one worker)
"""
import json
import sys
from pathlib import Path

import torch

import cost_pb
import pb

HERE = Path(__file__).resolve().parent
BAND = dict(a0=6, r=448, k=None, dtype=torch.float32, proj="qr", entry="pre")
CANDIDATES = {
    "band_6_448_pre": (dict(bandz=dict(BAND)), dict(band=dict(a0=6, r=448, k=None, proj="qr", entry="pre"))),
    "band_6_384_pre": (dict(bandz=dict(BAND, r=384)), dict(band=dict(a0=6, r=384, k=None, proj="qr", entry="pre"))),
    "band_5_448_pre": (dict(bandz=dict(BAND, a0=5)), dict(band=dict(a0=5, r=448, k=None, proj="qr", entry="pre"))),
    "band_5_512_pre": (dict(bandz=dict(BAND, a0=5, r=512)), dict(band=dict(a0=5, r=512, k=None, proj="qr", entry="pre"))),
    "band_4_512_pre": (dict(bandz=dict(BAND, a0=4, r=512)), dict(band=dict(a0=4, r=512, k=None, proj="qr", entry="pre"))),
    "band_5_448_pre_rr": (dict(bandz=dict(BAND, a0=5, init="rr")),
                          dict(band=dict(a0=5, r=448, k=None, proj="qr", entry="pre", init="rr"))),
    "band_6_448_pre_rr": (dict(bandz=dict(BAND, init="rr")),
                          dict(band=dict(a0=6, r=448, k=None, proj="qr", entry="pre", init="rr"))),
    "band_5_480_pre_rr": (dict(bandz=dict(BAND, a0=5, r=480, init="rr")),
                          dict(band=dict(a0=5, r=480, k=None, proj="qr", entry="pre", init="rr"))),
    "band_4_512_pre_rr": (dict(bandz=dict(BAND, a0=4, r=512, init="rr")),
                          dict(band=dict(a0=4, r=512, k=None, proj="qr", entry="pre", init="rr"))),
    "band_6_448_post": (dict(bandz=dict(BAND, entry="post")), dict(band=dict(a0=6, r=448, k=None, proj="qr", entry="post"))),
    "e3_-3.0_a": (dict(recv=(-3.0, "a")), dict(e3=dict(theta=-3.0, arm="a"))),
    "e3_-3.25_a": (dict(recv=(-3.25, "a")), dict(e3=dict(theta=-3.25, arm="a"))),
    "e3_-3.0_b": (dict(recv=(-3.0, "b")), dict(e3=dict(theta=-3.0, arm="b"))),
    "e2_3.0_restore": (dict(satrow=3.0, satrow_k4="restore"), dict(satrow=3.0)),
    "e2_3.0_sep": (dict(satrow=3.0, satrow_k4="sep"), dict(satrow=3.0)),
    "e2_2.75_sep": (dict(satrow=2.75, satrow_k4="sep"), dict(satrow=2.75)),
    "e2_2.5_sep": (dict(satrow=2.5, satrow_k4="sep"), dict(satrow=2.5)),
    "e2_2.25_sep": (dict(satrow=2.25, satrow_k4="sep"), dict(satrow=2.25)),
    "e2_2.0_sep": (dict(satrow=2.0, satrow_k4="sep"), dict(satrow=2.0)),
    "e2_2.5_restore": (dict(satrow=2.5, satrow_k4="restore"), dict(satrow=2.5)),
    "e4_4_-2.0": (dict(agecut=(4, -2.0)), dict(e4=dict(a_min=4, t_old=-2.0))),
    "e4_4_-1.5": (dict(agecut=(4, -1.5)), dict(e4=dict(a_min=4, t_old=-1.5))),
}


def stack_cfg(names, t=-2.25):
    cfg, ucfg = pb.matched(t), dict(t_row=t, t_col=t)
    for nm in names:
        ck, uk = CANDIDATES[nm]
        cfg.update(ck)
        ucfg.update(uk)
    return cfg, ucfg


def projected(raw, ucfg, l5=False):
    u = cost_pb.util(**ucfg)
    if l5:
        u = u * cost_pb.L5_METER
    return pb.GRADED_K3M * (raw / pb.K3M_RAW) * (u / pb.UTIL_K3M), u


if __name__ == "__main__":
    out_json = Path(sys.argv[1])
    t = float(sys.argv[2])
    names = sys.argv[3:]
    cfg, ucfg = stack_cfg(names, t)
    tag = f"t={t} " + " + ".join(names) if names else f"t={t} matched"
    out = pb.run_arms([(tag, cfg)], HERE / out_json, ref_per=None)
    r = out[tag]
    if "failed" not in r:
        adj, u = projected(r["raw"], ucfg)
        r.update(names=names, t=t, util=u, util_x=u / pb.UTIL_K3M, raw_x=r["raw"] / pb.K3M_RAW, projected=adj,
                 projected_l5=projected(r["raw"], ucfg, l5=True)[0])
        out[tag] = r
        (HERE / out_json).write_text(json.dumps(out, indent=1))
        print(f"   {tag}: raw x{r['raw_x']:.4f} util {100 * u:.2f}% projected {adj:.4e} (L5 {r['projected_l5']:.4e})")
