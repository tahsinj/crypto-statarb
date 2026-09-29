"""Panels and the point-in-time universe.

The raw frame built by quantlib.fetch is indexed by date and has a two-level
column index (symbol, field). Fields are 'price' (close), 'volume' (USD quote
volume) and, for Binance klines, 'taker' (taker-buy quote volume). Everything
downstream works with plain dates x symbols panels.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Stablecoins carry no momentum/reversal signal and would pollute cross-sectional
# ranks, so they are excluded from every universe.
STABLECOINS = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDD", "USDP", "GUSD",
    "FRAX", "LUSD", "USDE", "PYUSD", "EURT", "EURS", "VNST", "USTC", "CRVUSD",
    "USD0", "USDJ", "SUSD", "DOLA", "MIM", "USDX", "GHO",
}

# Fiat, gold-backed and newer dollar tokens that also trade against USDT. The
# research coin list (fetched in July 2026) let EUR, PAXG, XAUT, RLUSD, USD1
# and U through; the archive rebuild in notebook 09 drops them.
PEGGED = {
    "EUR", "GBP", "AUD", "AEUR", "EURI", "BKRW", "PAX", "PAXG", "XAUT", "UST",
    "USDS", "USDSB", "XUSD", "USD1", "RLUSD", "U", "BFUSD", "KGST",
}
# Tokenized US shares and ETFs, listed on Binance spot from June 2026 with a
# B suffix (NVDAB is Nvidia). Not crypto. Five of them (CRCLB, MSTRB, MUB,
# SNDKB, SPCXB) were in the research coin list. Notebooks 09 and 12 print any
# new B-suffixed listing so the list can be kept up to date.
TOKENIZED_STOCKS = {
    "AAOIB", "AAPLB", "ALABB", "AMATB", "AMDB", "AMZNB", "ARMB", "ASMLB", "ASTSB", "AVGOB",
    "AXTIB", "BABAB", "BEB", "BMNRB", "CBRSB", "COHRB", "COINB", "CRCLB", "CRDOB", "CRWVB",
    "DELLB", "DJTB", "DRAMB", "EWYB", "FLNCB", "GLWB", "GMEB", "GOOGLB", "GSB", "HOODB",
    "IBMB", "INTCB", "INTWB", "IRENB", "KORUB", "LITEB", "METAB", "MRVLB", "MSFTB", "MSTRB",
    "MUB", "MUUB", "MVLLB", "NBISB", "NFLXB", "NOKB", "NVDAB", "ORCLB", "PLTRB", "PYPLB",
    "QCOMB", "QNTB", "QQQB", "RKLBB", "SKHYB", "SMCIB", "SMHB", "SNDKB", "SNXXB", "SOXLB",
    "SOXSB", "SPCXB", "SPYB", "TQQQB", "TSLAB", "TSMB", "USARB", "WDCB",
}
WRAPPED = {"WBTC", "WETH", "WBETH", "BETH", "STETH", "WEETH"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
# Perp names that differ from the spot ticker by more than a size prefix.
PERP_ALIASES = {"LUNA2": "LUNA"}
# The research coin list: the 149 of the top 150 USDT pairs by 24h volume on
# 2026-07-06 that returned data, as cached in data/raw/binance_1d.pkl.zip.
# Eleven of them had in fact stopped trading before that day, from POLY
# (2022-10) to TON (2026-06-30). Notebook 00 fetches these by name, so a
# refetch asks for the same coins.
RESEARCH_COINS = (
    "AAVE", "ACE", "ADA", "AIGENSYN", "AIXBT", "ALGO", "ALICE", "ALLO", "APT", "AR", "ARB",
    "ARPA", "ASTER", "ATM", "ATOM", "AVAX", "BCH", "BEL", "BERA", "BIO", "BNB", "BONK",
    "BTC", "CAKE", "CELO", "CHZ", "CITY", "COCOS", "CRCLB", "CRV", "D", "DASH", "DEXE",
    "DOGE", "DOT", "DYDX", "EIGEN", "ENA", "EPIC", "ETC", "ETH", "ETHFI", "EUR", "FET",
    "FIL", "FLOKI", "FTM", "G", "GAL", "GALA", "GIGGLE", "GMT", "GRAM", "HBAR", "HEI",
    "HEMI", "HIGH", "HMSTR", "HOT", "ICP", "ID", "IMX", "INJ", "JST", "JTO", "KAITO",
    "KITE", "KSM", "LDO", "LINK", "LRC", "LTC", "LUNC", "MANTRA", "MEGA", "MINA", "MIRA",
    "MORPHO", "MOVR", "MSTRB", "MUB", "NEAR", "NEIRO", "NFP", "NOM", "OG", "OGN", "ONDO",
    "OP", "OPN", "ORDI", "PAXG", "PENDLE", "PENGU", "PEPE", "PHB", "POL", "POLY", "PUMP",
    "PYR", "PYTH", "RE", "RENDER", "RESOLV", "RIF", "RLUSD", "RNDR", "RUNE", "S", "SCRT",
    "SEI", "SENT", "SHIB", "SKY", "SNDKB", "SOL", "SPCXB", "STRK", "SUI", "SUN", "SYN",
    "TAO", "TIA", "TLM", "TON", "TRB", "TRUMP", "TRX", "TST", "U", "UNI", "USD1", "UTK",
    "VANRY", "VIRTUAL", "W", "WIF", "WLD", "WLFI", "XAUT", "XLM", "XPL", "XRP", "XTZ",
    "YFI", "ZAMA", "ZEC", "ZK", "ZRO",
)


def is_leveraged_token(base: str, bases: set[str]) -> bool:
    """Binance's old leveraged tokens: BTCUP, ETHDOWN, BNBBULL, BULL, BEAR...

    A name counts only if it is another listed base plus one of the suffixes,
    so JUP and SYRUP stay in. The fetch-time filter used for the research coin
    list (fetch.top_usdt_pairs) checked the suffix alone and dropped both.
    """
    if base in ("BULL", "BEAR"):
        return True
    return any(base.endswith(s) and base[: -len(s)] in bases for s in LEVERAGED_SUFFIXES)


def tradable_bases(bases: list[str]) -> list[str]:
    """Drop stablecoins, pegged assets, tokenized stocks, wrapped coins and leveraged tokens."""
    bset = set(bases)
    return sorted(b for b in bset if b not in STABLECOINS | PEGGED | WRAPPED | TOKENIZED_STOCKS
                  and not is_leveraged_token(b, bset))


def perp_to_spot(perps: list[str], spot: list[str]) -> dict[str, str]:
    """Map perp base names to the spot names they track.

    '1000SHIB' -> 'SHIB' and '1MBABYDOGE' -> 'BABYDOGE' (contract size
    prefixes), unless the prefixed name is itself a spot ticker, as 1000SATS
    is. Aliases in PERP_ALIASES are applied last. Perps with no spot match
    (index contracts such as BTCDOM) are left out.
    """
    spot_set = set(spot)
    out = {}
    for p in perps:
        name = PERP_ALIASES.get(p, p)
        if name not in spot_set:
            name = re.sub(r"^(1000000|100000|10000|1000|1M)(?=[A-Z])", "", name)
        if name in spot_set:
            out[p] = name
    return out


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a raw frame written by fetch.save_raw (.pkl, or .pkl.zip)."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            inner = next(n for n in z.namelist() if n.endswith(".pkl"))
            return pd.read_pickle(io.BytesIO(z.read(inner)))
    return pd.read_pickle(path)


def to_panels(
    raw: pd.DataFrame,
    drop_stablecoins: bool = True,
) -> dict[str, pd.DataFrame]:
    """Split the raw MultiIndex frame into price, dollar_volume and returns panels."""
    raw = raw.sort_index()
    price = raw.xs("price", axis=1, level=1).copy()
    volume = raw.xs("volume", axis=1, level=1).copy()

    if drop_stablecoins:
        keep = [c for c in price.columns if c not in STABLECOINS]
        price, volume = price[keep], volume[keep]

    price = price.sort_index(axis=1)
    volume = volume.reindex_like(price)

    fields = set(raw.columns.get_level_values(1))
    taker = None
    if "taker" in fields:
        taker = raw.xs("taker", axis=1, level=1).reindex_like(price)

    price = price.where(price > 0)  # non-positive prices are bad ticks
    returns = price.pct_change()

    out = {"price": price, "dollar_volume": volume, "returns": returns}
    if taker is not None:
        # Share of volume initiated by aggressive buyers; ~0.5 = balanced flow.
        out["taker_imbalance"] = (taker / volume).where(volume > 0)
    return out


def contract_size(perp: str, coin: str) -> float:
    """Coins per perp contract: 1000 for '1000SHIB' on SHIB, 1e6 for '1MBABYDOGE'."""
    prefix = perp[: len(perp) - len(coin)] if perp.endswith(coin) else ""
    if prefix == "1M":
        return 1e6
    return float(prefix) if prefix.isdigit() else 1.0


def archive_panels(
    raw_spot: pd.DataFrame,
    raw_perp: pd.DataFrame,
    funding: pd.DataFrame,
    perp_names: dict[str, str],
    max_basis: float = 0.2,
    max_break: int = 3,
) -> dict[str, pd.DataFrame]:
    """Panels for the every-pair universe built from the archive (notebooks 09 and 12).

    Spot panels as in to_panels, plus the daily high and low. Perp panels
    (perp_price, perp_high, perp_low, perp_returns, perp_dollar_volume) and
    funding are renamed to the spot coin each contract tracks (perp_names, from
    perp_to_spot). A coin that has had two contracts, such as LUNA before and
    after its 2022 relaunch (LUNA2), gets them joined by date. Returns are
    computed within each contract first, so a switch of contract never shows
    up as a return.

    A contract only counts as tracking its coin on days when its price is
    within ``max_basis`` of the spot price (after the contract size) and it
    traded at all. Some perps drift away from the spot ticker after a token
    migration on one side, and the archive keeps writing a frozen, zero-volume
    bar every day for a perp that has been delisted. On days that fail, all of
    the perp's data, funding included, is dropped, and a return also needs the
    day before to pass. ``panels['perp_valid']`` records which days passed.

    That rule decides which perps a strategy can pick. A position already held
    earns what its contract did, so the same panels without it end in
    ``_traded`` (perp_price, perp_high, perp_low, perp_returns, funding): each
    contract's own bars and funding on every day it traded, however far from
    spot. A traded return runs from the contract's last traded close; a break
    of more than ``max_break`` days, such as a delisting, starts it afresh.
    """
    panels = to_panels(raw_spot)
    for field in ["high", "low"]:
        panels[field] = raw_spot.xs(field, axis=1, level=1).reindex_like(panels["price"])

    idx = panels["price"].index
    px = raw_perp.xs("price", axis=1, level=1).reindex(idx)
    vol = raw_perp.xs("volume", axis=1, level=1).reindex(idx)
    valid = {}
    for perp, coin in perp_names.items():
        if perp in px.columns and coin in panels["price"].columns:
            ratio = px[perp] / (contract_size(perp, coin) * panels["price"][coin])
            valid[perp] = ((ratio - 1).abs() <= max_basis) & (vol[perp] > 0)
    valid = pd.DataFrame(valid, index=idx).fillna(False).astype(bool)
    frames = {
        "perp_price": px,
        "perp_high": raw_perp.xs("high", axis=1, level=1).reindex(idx),
        "perp_low": raw_perp.xs("low", axis=1, level=1).reindex(idx),
        "perp_returns": px.where(px > 0).pct_change(),
        "perp_dollar_volume": raw_perp.xs("volume", axis=1, level=1).reindex(idx),
        "funding": funding.reindex(idx),
    }
    for name, frame in frames.items():
        cols: dict[str, pd.Series] = {}
        for perp, coin in sorted(perp_names.items()):
            if perp in frame.columns and perp in valid.columns:
                ok = valid[perp] & valid[perp].shift(1, fill_value=False) if name == "perp_returns" else valid[perp]
                s = frame[perp].where(ok)
                cols[coin] = s if coin not in cols else cols[coin].combine_first(s)
        panels[name] = pd.DataFrame(cols, index=idx).sort_index(axis=1)
    panels["perp_valid"] = valid

    traded = ((vol > 0) & (px > 0)).reindex(columns=valid.columns, fill_value=False)
    close = px.reindex(columns=valid.columns).where(traded)
    frames = {
        "perp_price_traded": close,
        "perp_high_traded": frames["perp_high"].reindex(columns=valid.columns).where(traded),
        "perp_low_traded": frames["perp_low"].reindex(columns=valid.columns).where(traded),
        "perp_returns_traded": (close / close.ffill(limit=max_break).shift(1) - 1).where(traded),
        "funding_traded": frames["funding"].reindex(columns=valid.columns).where(traded),
    }
    for name, frame in frames.items():
        cols = {}
        for perp, coin in sorted(perp_names.items()):
            if perp in valid.columns:
                cols[coin] = frame[perp] if coin not in cols else cols[coin].combine_first(frame[perp])
        panels[name] = pd.DataFrame(cols, index=idx).sort_index(axis=1)
    return panels


def build_universe(
    panels: dict[str, pd.DataFrame],
    top_n: int | None = 100,
    min_adv_usd: float | None = None,
    adv_window: int = 30,
    min_history: int = 30,
) -> pd.DataFrame:
    """Build a point-in-time tradable universe (boolean dates x symbols).

    A coin is eligible on date t if, using only data through t, it has at least
    min_history trailing prices and its trailing adv_window median dollar-volume
    clears the liquidity bar (an absolute floor and/or top_n by ADV). Trailing
    medians (not point values) avoid selecting on the same-day volume spike that
    tends to accompany a return.
    """
    price = panels["price"]
    dvol = panels["dollar_volume"]

    # Shift by 1 so date t only sees volume through t-1: no look-ahead.
    adv = dvol.rolling(adv_window, min_periods=max(5, adv_window // 2)).median().shift(1)
    has_history = price.notna().rolling(min_history, min_periods=min_history).sum().shift(1) >= min_history

    eligible = adv.notna() & has_history
    if min_adv_usd is not None:
        eligible &= adv >= min_adv_usd

    if top_n is not None:
        ranks = adv.where(eligible).rank(axis=1, ascending=False, method="first")
        eligible &= ranks <= top_n

    return eligible.fillna(False)


def coverage_report(panels: dict[str, pd.DataFrame], universe: pd.DataFrame) -> pd.DataFrame:
    """Per-date diagnostics: coins with a price, and tradable-universe size."""
    price = panels["price"]
    return pd.DataFrame(
        {
            "coins_with_price": price.notna().sum(axis=1),
            "universe_size": universe.sum(axis=1),
        }
    )


def market_return(panels: dict[str, pd.DataFrame], universe: pd.DataFrame | None = None) -> pd.Series:
    """Equal-weight crypto-market daily return, used as a benchmark alongside BTC."""
    rets = panels["returns"]
    if universe is not None:
        rets = rets.where(universe.shift(1).fillna(False))
    return rets.mean(axis=1)
