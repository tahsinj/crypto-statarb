"""Performance metrics, significance tests, and factor decomposition.

One implementation of each statistic, imported everywhere, so the numbers stay
consistent across notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

TRADING_DAYS = 365


def _nw_lags(n: int) -> int:
    """Newey-West truncation lag via the 4*(n/100)**(2/9) rule."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


# --- core risk / return ---
def ann_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric annualised return."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    return (1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0


def ann_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    r = returns.dropna() - rf / periods_per_year
    if r.std() == 0 or r.empty:
        return np.nan
    return r.mean() / r.std() * np.sqrt(periods_per_year)


def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    r = returns.dropna() - rf / periods_per_year
    downside = r[r < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return r.mean() / downside * np.sqrt(periods_per_year)


def drawdown_curve(returns: pd.Series) -> pd.Series:
    eq = (1.0 + returns.fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return drawdown_curve(returns).min()


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest run (in days) spent below a prior equity high-water mark."""
    dd = drawdown_curve(returns)
    underwater = dd < 0
    longest = run = 0
    for flag in underwater:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def calmar(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(returns))
    return ann_return(returns, periods_per_year) / mdd if mdd > 0 else np.nan


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    return (r > 0).mean() if len(r) else np.nan


# --- significance ---
def sharpe_tstat(returns: pd.Series, hac: bool = False, max_lags: int | None = None) -> float:
    """t-stat for Sharpe != 0. Roughly Sharpe * sqrt(years).

    With hac=True the t-stat uses a Newey-West standard error instead of the iid
    std/sqrt(n). Daily returns are autocorrelated (momentum trends, pairs
    hysteresis), so the iid t-stat overstates significance; quote the HAC one.
    """
    r = returns.dropna()
    n = len(r)
    if n < 2 or r.std() == 0:
        return np.nan
    if not hac:
        return r.mean() / r.std() * np.sqrt(n)
    lags = _nw_lags(n) if max_lags is None else max_lags
    fit = sm.OLS(r.to_numpy(), np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def sharpe_pvalue(returns: pd.Series, hac: bool = False, max_lags: int | None = None) -> float:
    t = sharpe_tstat(returns, hac=hac, max_lags=max_lags)
    n = len(returns.dropna())
    if np.isnan(t) or n < 2:
        return np.nan
    return 2 * (1 - stats.t.cdf(abs(t), df=n - 1))


def deflated_sharpe(returns: pd.Series, n_trials: int = 1) -> float:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014).

    Probability that the observed Sharpe beats the best Sharpe that n_trials
    zero-skill strategies would show by luck, allowing for skew and fat tails.
    Between 0 and 1; 0.95 is the usual bar.
    """
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std()  # per-period (non-annualised)
    sk = stats.skew(r)
    ku = stats.kurtosis(r, fisher=False)

    # SE of the per-period Sharpe estimate, adjusted for skew and kurtosis (Lo, 2002).
    sr_std = np.sqrt((1 - sk * sr + (ku - 1) / 4.0 * sr**2) / (n - 1))
    if sr_std == 0:
        return np.nan

    # Expected best Sharpe out of n_trials strategies whose true Sharpe is zero.
    # The bracket approximates the expected maximum of n_trials standard
    # normals; multiplying by sr_std puts sr0 in per-period Sharpe units. Using
    # sr_std for the spread across trials is a simplification: under the null,
    # each trial's estimate has roughly that standard error.
    if n_trials > 1:
        emc = 0.5772156649  # Euler-Mascheroni
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr0 = sr_std * ((1 - emc) * z1 + emc * z2)
    else:
        sr0 = 0.0

    return float(stats.norm.cdf((sr - sr0) / sr_std))


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    method: str = "block",
    block_size: int | None = None,
    periods_per_year: int = TRADING_DAYS,
):
    """Bootstrap CI for the annualised Sharpe ratio.

    method="block" (default) resamples contiguous blocks of length block_size
    (default ~n**(1/3)) rather than individual days, so the CI respects serial
    dependence in daily returns; an iid resample understates the uncertainty of
    an autocorrelated Sharpe. method="iid" keeps the plain iid resample.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 30:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)

    if method == "iid":
        boot = np.empty(n_boot)
        for i in range(n_boot):
            sample = rng.choice(r, size=n, replace=True)
            sd = sample.std()
            boot[i] = sample.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan
    elif method == "block":
        b = block_size or max(2, int(round(n ** (1.0 / 3.0))))
        n_blocks = int(np.ceil(n / b))
        starts_pool = n - b + 1  # valid block start positions
        boot = np.empty(n_boot)
        for i in range(n_boot):
            starts = rng.integers(0, starts_pool, size=n_blocks)
            sample = np.concatenate([r[s:s + b] for s in starts])[:n]
            sd = sample.std()
            boot[i] = sample.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan
    else:
        raise ValueError(f"unknown method {method!r}; use 'block' or 'iid'")

    lo, hi = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


# --- alpha / beta ---
def alpha_beta(
    returns: pd.Series,
    *factors: pd.Series,
    names: list[str] | None = None,
    hac: bool = True,
    max_lags: int | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """OLS of strategy returns on one or more factor return series.

    Returns annualised alpha, per-factor betas, R², and the residual information
    ratio. Pass BTC and/or the equal-weight crypto market as factors. hac=True
    (default) uses Newey-West standard errors so the alpha t-stat is robust to
    autocorrelation and heteroskedasticity in daily returns.
    """
    fnames = list(names) if names is not None else [f"x{i}" for i in range(len(factors))]
    df = pd.concat(
        [returns.rename("y")] + [f.rename(nm) for f, nm in zip(factors, fnames)],
        axis=1,
    ).dropna()
    if df.empty:
        return {}
    y = df["y"].to_numpy()
    X = sm.add_constant(df.drop(columns="y").to_numpy())  # const first, then factors in order
    n = len(y)

    if hac:
        lags = _nw_lags(n) if max_lags is None else max_lags
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    else:
        fit = sm.OLS(y, X).fit()
    coef = fit.params
    tvals = fit.tvalues

    # IR = annualised alpha / annualised residual vol. Use the intercept, not
    # resid.mean(): the latter is ~0 by OLS construction and makes the IR zero.
    resid_vol = fit.resid.std()
    info_ratio = coef[0] / resid_vol * np.sqrt(periods_per_year) if resid_vol > 0 else np.nan
    out = {
        "alpha_daily": coef[0],
        "alpha_ann": coef[0] * periods_per_year,
        "alpha_tstat": tvals[0],
        "r_squared": fit.rsquared,
        "resid_info_ratio": info_ratio,
    }
    for name, b in zip(fnames, coef[1:]):
        out[f"beta_{name}"] = b
    return out


def treynor_mazuy(
    returns: pd.Series,
    market: pd.Series,
    freq: str = "W",
    periods_per_year: int = 52,
    max_lags: int = 4,
) -> dict:
    """Market-timing regression r = a + b*m + c*m**2 (Treynor and Mazuy, 1966).

    A strategy that is long when the market rises and short when it falls, such
    as trend following, has a convex payoff (c > 0). A plain regression on m
    averages its changing beta to about zero and books the timing gain as
    alpha; this one separates the two. Returns are compounded to `freq`
    (weekly by default) because a trend book's exposure changes slowly, and
    the standard errors are Newey-West.
    """
    df = pd.concat([returns.rename("r"), market.rename("m")], axis=1, sort=True).dropna()
    agg = (1.0 + df).resample(freq).prod() - 1.0
    X = sm.add_constant(pd.DataFrame({"m": agg["m"], "m2": agg["m"] ** 2}))
    fit = sm.OLS(agg["r"], X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    return {
        "alpha_ann": fit.params["const"] * periods_per_year,
        "alpha_tstat": fit.tvalues["const"],
        "beta": fit.params["m"],
        "timing": fit.params["m2"],
        "timing_tstat": fit.tvalues["m2"],
    }


# --- one-call summary ---
def summary(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    n_trials: int = 1,
    name: str = "strategy",
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Full metric panel as a single labelled Series (one column per strategy)."""
    out = {
        "Ann. Return": ann_return(returns, periods_per_year),
        "Ann. Vol": ann_vol(returns, periods_per_year),
        "Sharpe": sharpe(returns, periods_per_year=periods_per_year),
        "Sortino": sortino(returns, periods_per_year=periods_per_year),
        "Calmar": calmar(returns, periods_per_year),
        "Max Drawdown": max_drawdown(returns),
        "Max DD Days": max_drawdown_duration(returns),
        "Hit Rate": hit_rate(returns),
        "Sharpe t-stat": sharpe_tstat(returns),
        "Sharpe p-value": sharpe_pvalue(returns),
        "Deflated Sharpe": deflated_sharpe(returns, n_trials=n_trials),
    }
    if turnover is not None:
        avg_to = turnover.reindex(returns.index).mean()
        out["Avg Turnover"] = avg_to
        out["Holding Days"] = (1.0 / avg_to) if avg_to and avg_to > 0 else np.nan
    if benchmark is not None:
        ab = alpha_beta(returns, benchmark, names=["benchmark"], periods_per_year=periods_per_year)
        out["Alpha (ann)"] = ab.get("alpha_ann", np.nan)
        out["Beta"] = ab.get("beta_benchmark", np.nan)
        out["Info Ratio"] = ab.get("resid_info_ratio", np.nan)
    return pd.Series(out, name=name)
