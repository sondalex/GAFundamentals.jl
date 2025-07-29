"""
Download stock price and financial fundamentals


For compatibility with Parquet.jl make sure to install
pyarrow<20.0.0
"""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

PRICE = "Close"


def iso8601(x: str) -> datetime:
    return datetime.fromisoformat(x)


def cs_list(x: str, sep=",") -> List[str]:
    return x.split(sep=sep)
    pass


def register_fundamentals(subparsers):
    parser = subparsers.add_parser("fundamentals")
    parser.add_argument("symbols", type=cs_list)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-date", type=iso8601, required=True)
    parser.add_argument("--end-date", type=iso8601, required=True)
    return parser


def register_price(subparsers):
    parser = subparsers.add_parser("price")
    parser.add_argument("symbols", type=cs_list)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-date", type=iso8601, required=True)
    parser.add_argument("--end-date", type=iso8601, required=True)


def cli() -> ArgumentParser:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="")
    register_fundamentals(subparsers)
    register_price(subparsers)
    return parser


# %%
def download_fundamentals(symbols: List[str], start_date: datetime, end_date: datetime):
    """

    Example
    -------
    >>> download_fundamentals(["AAPL"], datetime(2015, 1, 1), datetime(2020, 1, 1))


    Warning
    -------
    This is mocked implementation, the data is randomly generated.
    For real data: `Financial Modeling Prep https://site.financialmodelingprep.com/developer/docs/stable#income-statement`_
    could be a solution

    ESG can be downloaded from:
        https://site.financialmodelingprep.com/developer/docs/stable#esg-search

    ValuationRatios can be downloaded from (Key Metrics):
        https://site.financialmodelingprep.com/developer/docs/stable#metrics-ratios

    Health (Key Metrics and Financial Ratios):
       https://site.financialmodelingprep.com/developer/docs/stable#metrics-ratios
    """
    T = (end_date.year - start_date.year) + 1  # inclusive of end date year
    columns = pd.MultiIndex.from_tuples(
        [
            ("ESG", "environmentalScore"),
            ("ESG", "socialScore"),
            ("ESG", "governanceScore"),
            ("Valuation", "priceToEarningsRatio"),
            ("Valuation", "priceToBookRatio"),
            ("Valuation", "priceToSalesRatio"),
            ("Health", "currentRatio"),
            ("Health", "debtToEquityRatio"),
            ("Health", "returnOnAssets"),
        ]
    )
    data = []
    for symbol in symbols:
        values = np.random.uniform(size=(T, len(columns)))
        data.append(values)
    data = np.vstack(data)
    index = pd.MultiIndex.from_tuples(
        [
            (symbol, datetime(start_date.year + t, 1, 1))
            for symbol in symbols
            for t in range(T)
        ]
    )
    df = pd.DataFrame(data, columns=columns, index=index)
    df.index.names = ("Ticker", "Year")
    return df


def multicolumn_df_to_singlecolumn(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected multiindex columns")
    columns = df.columns
    if not len(df.columns.levels) == 2:
        raise ValueError("Expected two levels multiindex columns")
    df.columns = df.columns.levels[1]
    return df, {tuple_[1]: tuple_[0] for tuple_ in columns}


def write(df: pd.DataFrame, output: Path):
    df.to_parquet(output, index=False)


def download_price(
    symbols: List[str], start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    data = yf.download(
        symbols, start=start_date.date().isoformat(), end=end_date.date().isoformat()
    )
    data = (
        data.stack(dropna=False).reorder_levels([1, 0])[[PRICE]].reset_index(drop=False)
    )
    return data


def main():
    parser = cli()
    args = parser.parse_args()
    match args.command:
        case "fundamentals":
            fundamentals = download_fundamentals(
                args.symbols, args.start_date, args.end_date
            )
            fundamentals, attrs = multicolumn_df_to_singlecolumn(fundamentals)
            fundamentals.attrs = {"columns": attrs}
            write(fundamentals.reset_index(drop=False), args.output)
        case "price":
            prices = download_price(args.symbols, args.start_date, args.end_date)
            write(prices, args.output)


if __name__ == "__main__":
    import sys

    sys.exit(main())
