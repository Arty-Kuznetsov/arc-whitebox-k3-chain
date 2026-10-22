// Run in the browser console (or the built-in browser's javascript tool) on
// https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/<id>
// Returns per-MLP grader telemetry summary for all 100 MLPs (public + private; timing only, scores of private are sealed).
(() => {
  const bar = document.querySelector('.tb-bar');
  let el = bar, sub = null;
  for (let k = 0; k < 8 && el && !sub; k++) {
    const key = Object.keys(el).find(x => x.startsWith('__reactFiber'));
    if (key) {
      let f = el[key];
      for (let j = 0; j < 12 && f; j++) {
        const p = f.memoizedProps;
        if (p && (p.row || p.mlp || p.data || p.item)) { sub = (p.row || p.mlp || p.data || p.item).submission; break; }
        f = f.return;
      }
    }
    el = el.parentElement;
  }
  const ev = sub.evaluation, pm = ev.results.per_mlp;
  const get = k => pm.map(p => p.telemetry && p.telemetry[k]).filter(x => x != null);
  const stats = a => ({ n: a.length, mean: a.reduce((x, y) => x + y, 0) / a.length, min: Math.min(...a), max: Math.max(...a) });
  return JSON.stringify({
    score: sub.score, secondary: sub.scoreSecondary, progress: ev.progress,
    residual_wall_time_s: stats(get('residual_wall_time_s')),
    predict_wall_s: stats(get('participant_predict_wall_s')),
    residual_exhausted: get('residual_wall_time_exhausted').filter(x => x).length,
    time_exhausted: get('time_exhausted').filter(x => x).length,
  });
})()
