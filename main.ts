// gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate
// at a case's target, as five quantiles on the logit scale.
//
// The model, in one paragraph: an earn rate is earners divided by players
// tracked, so it is forecast by forecasting the two counts separately.
// Both counts are fitted, on the log scale, against log time since the
// game's release (launch-week growth is fast then slows, so log time keeps
// a two-day fit from exploding over a four-day horizon), using only the
// distinct page states in the case's history, because Exophase refreshes
// its counts in jumps and the reads between refreshes are the same page
// frozen. Where a fit cannot be identified -- the page never moved in the
// history -- a launch-week prior stands in: tracked players grow fast
// after release, and each trophy's earners grow with the player count to
// a power that depends on the trophy's tier (platinum slowest, bronze
// fastest), so a pending refresh pulls most rates down by dilution and
// the median sits below the last read. The practice cases calibrate the
// size of that pull: the house's published practice aggregates for its
// own baseline (loss about 1.51, typical miss about 0.68) are matched by
// earners growing about a quarter while tracked players grow sixfold at
// the refresh, so the priors are set near that scale. The spread adds
// three honest uncertainties: the random-walk drift over the horizon, the fit's own residuals stretched by how far it
// extrapolates, and the size of the jumps past refreshes made (or a prior
// jump size where none was seen). Quantiles come from the standard normal
// on the logit scale. Everything is a pure function of the case file, so
// two runs write the same bytes; only Node's standard library is used.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Random-walk spread, in logit units per square-root day (the house baseline's). */
const RW_SIGMA = 0.5;
/** Launch-week tracked-player growth prior, log units per hour, used only when
 *  the history shows no player growth to fit (the page never moved). */
const PRIOR_PLAYER_GROWTH_PER_HOUR = 0.35;
/** Spread of a refresh jump, in logit units, used only when the history shows
 *  no refresh to measure one from. Set from the practice evaluation: the
 *  wider prior over-covered the answers, so it was tightened to 0.30. */
const PRIOR_JUMP_SIGMA = 0.3;
/** Spread of a fit that cannot be identified from the history. Set from
 *  the practice evaluation alongside the jump prior. */
const PRIOR_FIT_SIGMA = 0.22;
/** Narrowest and widest total spread allowed, in logit units. */
const SIGMA_MIN = 0.25;
const SIGMA_MAX = 1.3;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes: string;
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampPct = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Acklam's rational approximation to the standard normal quantile. */
function normQuantile(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285984469534e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239e0];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0, -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0];
  const plow = 0.02425;
  const phigh = 1 - plow;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

function zFor(level: number): number {
  const i = LEVELS.indexOf(level);
  return i >= 0 ? Z[i] : normQuantile(level);
}

/** Ordinary least squares of y on x. Returns slope, intercept, residual
 *  standard deviation, the x mean and the sum of squared x deviations. */
function ols(xs: number[], ys: number[]): { a: number; b: number; resid: number; xMean: number; sxx: number; n: number } {
  const n = xs.length;
  const xMean = xs.reduce((s, v) => s + v, 0) / n;
  const yMean = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - xMean) * (xs[i] - xMean);
    sxy += (xs[i] - xMean) * (ys[i] - yMean);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = yMean - b * xMean;
  let rss = 0;
  for (let i = 0; i < n; i++) rss += (ys[i] - (a + b * xs[i])) ** 2;
  const resid = n > 2 ? Math.sqrt(rss / (n - 2)) : 0;
  return { a, b, resid, xMean, sxx, n };
}

/** How fast a trophy's earners grow with the player count, when the history
 *  cannot show it: a tier gradient, platinum slowest and bronze fastest.
 *  The practice calibration showed short-horizon earner growth is nearly
 *  uniform across bronze trophies (story or not), so bronze takes one
 *  value; where the history does move, the per-trophy fit overrides this. */
function gammaPrior(t: Trophy): number {
  if (t.tier === 'platinum') return 0.05;
  if (t.tier === 'gold') return 0.1;
  if (t.tier === 'silver') return 0.15;
  return 0.2;
}

function samePage(a: Read, b: Read): boolean {
  if (a.players_tracked !== b.players_tracked) return false;
  if (a.rates.length !== b.rates.length) return false;
  for (let i = 0; i < a.rates.length; i++) if (a.rates[i] !== b.rates[i]) return false;
  return true;
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  const levels = c.levels.length === LEVELS.length && c.levels.every((t, j) => t === LEVELS[j])
    ? LEVELS
    : c.levels;
  const zs = levels.map(zFor);
  const last = c.history[c.history.length - 1];
  if (last === undefined) throw new Error(`${c.id} has no history`);
  const horizonDays = c.horizon_minutes / 1440;
  const horizonHours = c.horizon_minutes / 60;
  const checkpoint = Date.parse(c.at);
  const target = Date.parse(c.target);
  const release = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const origin = Number.isFinite(release) ? release : Date.parse(c.history[0].at);
  // Log time since release, in units of a day: launch-week growth is fast
  // then slows, so this clock keeps a two-day fit from exploding over a
  // four-day horizon.
  const xOf = (ms: number): number => Math.log(1 + Math.max(0, (ms - origin) / 3_600_000) / 24);
  const xTarget = xOf(target);
  const xLast = xOf(Date.parse(last.at));

  // Distinct page states, oldest first: the reads between Exophase's
  // refreshes repeat the same page, so only the states carry information.
  const states: Read[] = [];
  for (const h of [...c.history].sort((p, q) => Date.parse(p.at) - Date.parse(q.at))) {
    if (states.length === 0 || !samePage(states[states.length - 1], h)) states.push(h);
  }
  const changeTimes = states.map((s) => Date.parse(s.at));
  // Hours the page has been frozen at the checkpoint: the pending refresh
  // has the whole frozen window's growth to catch up on.
  const lastChange = changeTimes.length > 0 ? changeTimes[changeTimes.length - 1] : Date.parse(c.history[0].at);
  const frozenHours = Math.max(0, (checkpoint - lastChange) / 3_600_000);

  // --- Players: fit log players on log time over the distinct states. ---
  const pXs = states.map((s) => xOf(Date.parse(s.at)));
  const pYs = states.map((s) => Math.log(Math.max(1, s.players_tracked)));
  const pVar = pYs.length > 1 ? Math.max(...pYs) - Math.min(...pYs) : 0;
  let pFit = pYs.length >= 2 ? ols(pXs, pYs) : null;
  if (pFit !== null) pFit = { ...pFit, b: clamp(pFit.b, 0, 4) };
  const pIdentified = pFit !== null && pVar > 0.01 && pFit.n >= 2;
  let logPTarget: number;
  let pResid: number;
  if (pIdentified && pFit !== null) {
    logPTarget = pFit.a + pFit.b * xTarget;
    pResid = pFit.resid;
  } else {
    // No player growth to fit: the prior says the tracked count grows
    // through the frozen window and the horizon at launch-week pace.
    logPTarget = Math.log(Math.max(1, last.players_tracked)) + PRIOR_PLAYER_GROWTH_PER_HOUR * (frozenHours + horizonHours);
    pResid = PRIOR_FIT_SIGMA;
  }
  const pRatio = Math.exp(logPTarget - Math.log(Math.max(1, last.players_tracked)));

  // --- Refresh jumps: the logit moves past refreshes made, pooled over
  //     trophies; a prior where no refresh was ever seen. ---
  const jumps: number[] = [];
  for (let s = 1; s < states.length; s++) {
    for (let i = 0; i < c.trophies.length; i++) {
      const r0 = states[s - 1].rates[i];
      const r1 = states[s].rates[i];
      if (r0 === null || r1 === null) continue;
      if (r0 <= 0 || r0 >= 100 || r1 <= 0 || r1 >= 100) continue;
      jumps.push(logit(r1) - logit(r0));
    }
  }
  let jumpSigma = PRIOR_JUMP_SIGMA;
  if (jumps.length >= 10) {
    const m = jumps.reduce((s, v) => s + v, 0) / jumps.length;
    const sd = Math.sqrt(jumps.reduce((s, v) => s + (v - m) * (v - m), 0) / jumps.length);
    jumpSigma = clamp(sd, 0.2, 1.0);
  }

  const sigmaRw = RW_SIGMA * Math.sqrt(horizonDays);

  const rows = c.trophies.map((t, i) => {
    const rLast = last.rates[i] === null || last.rates[i] === undefined ? 0 : (last.rates[i] as number);
    // A trophy essentially everyone has: its rate can only drift down a
    // little as new players join, so forecast near the top with a wide band.
    if (rLast >= 99.5) {
      const m = 98.5;
      const sigma = clamp(Math.sqrt(sigmaRw * sigmaRw + 0.8 * 0.8), SIGMA_MIN, SIGMA_MAX);
      return { id: t.id, q: zs.map((z) => round4(clampPct(invLogit(logit(m) + z * sigma)))) };
    }
    const eLast = Math.max(0.5, (rLast / 100) * Math.max(1, last.players_tracked));
    // Fit log earners on log time over the states that list this trophy.
    const eXs: number[] = [];
    const eYs: number[] = [];
    for (const s of states) {
      const r = s.rates[i];
      if (r === null || r === undefined) continue;
      eXs.push(xOf(Date.parse(s.at)));
      eYs.push(Math.log(Math.max(0.5, (r / 100) * Math.max(1, s.players_tracked))));
    }
    const eVar = eYs.length > 1 ? Math.max(...eYs) - Math.min(...eYs) : 0;
    const eFit = eXs.length >= 2 ? ols(eXs, eYs) : null;
    const eIdentified = eFit !== null && eVar > 0.01 && eFit.n >= 3;
    let logETarget: number;
    let eResid: number;
    let leverage = 1;
    if (eIdentified && eFit !== null) {
      const b = clamp(eFit.b, 0, 4);
      logETarget = eFit.a + b * xTarget;
      eResid = eFit.resid;
      const lev = eFit.sxx > 1e-12 ? 1 + (xTarget - eFit.xMean) * (xTarget - eFit.xMean) / eFit.sxx : 1;
      leverage = clamp(Math.sqrt(lev), 1, 3);
    } else {
      // No earner trend to fit: earners grow with the player count to the
      // power the trophy's kind allows, so the rate is diluted by the rest.
      const g = gammaPrior(t);
      logETarget = Math.log(eLast) + g * Math.log(pRatio);
      eResid = PRIOR_FIT_SIGMA;
    }
    const rate = clampPct((Math.exp(logETarget) / Math.exp(logPTarget)) * 100);
    const sigmaFit = clamp(eResid * leverage, 0, 1.0);
    const sigmaP = clamp(pResid * leverage, 0, 1.0);
    const sigma = clamp(Math.sqrt(sigmaRw * sigmaRw + sigmaFit * sigmaFit + sigmaP * sigmaP + jumpSigma * jumpSigma), SIGMA_MIN, SIGMA_MAX);
    return { id: t.id, q: zs.map((z) => round4(clampPct(invLogit(logit(rate) + z * sigma)))) };
  });
  return { trophies: rows };
}

/** The house baseline's forecast: last read's rates, random-walk spread.
 *  Only a fallback, so a case never crashes the run. */
function baselineForecast(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  const last = c.history[c.history.length - 1];
  const spread = RW_SIGMA * Math.sqrt(c.horizon_minutes / 1440);
  const zs = c.levels.map(zFor);
  return {
    trophies: c.trophies.map((t, i) => {
      const r = last.rates[i] === null || last.rates[i] === undefined ? 0 : (last.rates[i] as number);
      return { id: t.id, q: zs.map((z) => round4(clampPct(invLogit(logit(r) + z * spread)))) };
    }),
  };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  let out: { trophies: Array<{ id: number; q: number[] }> };
  try {
    out = forecastCase(c);
  } catch {
    out = baselineForecast(c);
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: out.trophies })}\n`);
}
