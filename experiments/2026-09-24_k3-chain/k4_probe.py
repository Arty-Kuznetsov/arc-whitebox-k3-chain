import sys, torch
exec(open("step_check.py").read().split('print(f"n={n} depth={depth}")')[0])
for l in range(depth - 1):
    pre, act, nxt = Kb[f"pre{l}"], Kb[f"act{l}"], Kb[f"pre{l+1}"]
    mu, C = pre[1].core, pre[2].core
    if 3 in pre:
        t, D = pre[3].get_dslice((3,)), offdiag(pre[3].get_dslice((2, 1)))
    else:
        t, D = torch.zeros(n), torch.zeros(n, n)
    K4d, K22 = k4_slices(pre[4]) if 4 in pre else (torch.zeros(n), torch.zeros(n, n))
    _, _, _, _, _, K4p, K22p = my_step(mu, C, t, D, K4d, K22)
    L2 = float(K4p.sum() + K22p.sum())
    a = act[4]
    m_act = a.metric if a.metric.ndim > 0 else a.metric
    s4, s22 = k4_slices(a)
    s4n, s22n = k4_slices(nxt[4])
    print(l, "core", float(a.core), "L2", L2, "core/L2", float(a.core) / L2, "r", a.r,
          "metric act", tuple(a.metric.shape), float(a.metric.reshape(-1)[0]),
          "| act slices (4,) mean", float(s4.mean()), "(2,2) mean", float(s22.sum() / (n * n - n)),
          "| next pre core", float(nxt[4].core), "metric", float(nxt[4].metric.reshape(-1)[0]),
          "(4,)", float(s4n.mean()), "(2,2)", float(s22n.sum() / (n * n - n)))
