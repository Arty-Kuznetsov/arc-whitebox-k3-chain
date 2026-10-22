#!/usr/bin/env python3
# Usage: python3 scripts/poll_sub.py <submission id>   (curl only, no browser, no key: the page embeds its evaluation)
"""Poll an AIcrowd submission page until it has a score; print the telemetry summary (public page, no key)."""
import re, html, json, sys, time, subprocess, statistics as st
sid = sys.argv[1]
url = f"https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{sid}"
last = None
for it in range(240):   # up to ~4 h (queued submissions have no evaluation yet)
    s = subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", url], capture_output=True, text=True).stdout
    props = re.findall(r'data-react-props="([^"]*)"', s)
    big = [p for p in props if len(p) > 100000]
    if big:
        d = json.loads(html.unescape(big[0]))
        sub = d['props']['data']['submission']; ev = sub.get('evaluation') or {}; pm = (ev.get('results') or {}).get('per_mlp') or []
        prog = ev.get('progress')
        if prog != last:
            print(time.strftime('%H:%M:%S'), prog, flush=True); last = prog
        if sub.get('score') is not None or (prog and prog.get('mlps_completed', 0) + prog.get('mlps_failed', 0) >= 100):
            done = [x for x in pm if x.get('telemetry') and x['telemetry'].get('participant_predict_wall_s') is not None]
            if not done:   # failed before any network graded (e.g. the smoke test): print what the page knows
                res = ev.get('results') or {}
                print(json.dumps(dict(score=sub.get('score'), status=sub.get('status'), message=sub.get('gradingMessage'),
                      smoke=res.get('smoke_test_result') or ev.get('smoke_test_result'), error=ev.get('error') or res.get('error'),
                      log=ev.get('gradingLogUrl') or sub.get('gradingLogUrl'), progress=prog), ensure_ascii=False), flush=True)
                json.dump(sub, open(f'/tmp/sub_{sid}.json', 'w')); break
            w = sorted(x['telemetry']['participant_predict_wall_s'] for x in done)
            r = sorted(x['telemetry']['residual_wall_time_s'] for x in done)
            fl = [x['telemetry']['flops_used'] for x in done]
            print(json.dumps(dict(score=sub.get('score'), secondary=sub.get('scoreSecondary'), n=len(done),
                  wall_median=w[len(w)//2], wall_p90=w[int(0.9*len(w))], wall_max=w[-1], wall_top5=[round(x,1) for x in w[-5:]],
                  resid_median=r[len(r)//2], resid_max=r[-1], flops_mean=sum(fl)/len(fl), util=sum(fl)/len(fl)/2**41,
                  time_exh=sum(bool(x['telemetry'].get('time_exhausted')) for x in done),
                  resid_exh=sum(bool(x['telemetry'].get('residual_wall_time_exhausted')) for x in done),
                  failed=[(x['mlp_index'], x.get('state'), x.get('error_code')) for x in pm if x.get('state') not in ('completed',)])), flush=True)
            json.dump(d['props']['data']['submission'], open(f'/tmp/sub_{sid}.json', 'w'))
            break
    time.sleep(60)
