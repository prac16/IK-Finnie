"""Validate CSV/XLSX uploads; all valuations come from the file, never a quote API."""
from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from zipfile import ZipFile, BadZipFile

from openpyxl import load_workbook
from agents.portfolio.analysis import Holding, Portfolio, PortfolioAnalysisAgent

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ROWS = 10000
ALIASES = {
    "sector": "sector", "sector_source": "sector_source",
    "ticker": "ticker", "asset_type": "asset_type", "asset_class": "asset_class",
    "market_value_usd": "market_value_usd", "weight_%": "weight_pct", "weight_pct": "weight_pct",
    "portfolio": "portfolio", "portfolio_id": "portfolio_id", "as_of_date": "as_of_date",
    "age": "age", "risk": "risk_tolerance", "risk_tolerance": "risk_tolerance",
    "goal": "goal", "analysis_focus": "analysis_focus", "currency": "currency",
}


def _number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{label}: expected a number") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label}: expected a finite nonnegative number")
    return result


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def load_holdings(content: bytes, filename: str) -> tuple[Portfolio, ...]:
    """Read one CSV or the Holdings sheet of an XLSX. Accept title/snake_case headers."""
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError("Holdings file must be nonempty and at most 10 MB")
    suffix = Path(filename).suffix.lower()
    profiles = {}
    if suffix == ".csv":
        try:
            rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        except (UnicodeDecodeError, csv.Error):
            raise ValueError("CSV must contain valid UTF-8 comma-separated data") from None
    elif suffix == ".xlsx":
        try:
            with ZipFile(io.BytesIO(content)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 50 * 1024 * 1024:
                    raise ValueError("Expanded XLSX exceeds 50 MB")
            workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=False)
            try:
                sheet = workbook["Holdings"] if "Holdings" in workbook.sheetnames else workbook.active
                rows = []
                for row in sheet.iter_rows():
                    if any(cell.data_type == "f" for cell in row):
                        raise ValueError("Holdings must contain values, not Excel formulas; paste values before uploading")
                    rows.append([cell.value for cell in row])
                    if len(rows) > MAX_ROWS + 1:
                        raise ValueError(f"Holdings exceed {MAX_ROWS} rows")
                if "Portfolios" in workbook.sheetnames:
                    for row in workbook["Portfolios"].iter_rows(min_row=2, values_only=True):
                        if row[0] is not None:
                            if len(profiles) >= MAX_ROWS:
                                raise ValueError("Too many portfolio profiles")
                            profiles[_text(row[0])] = row
            finally:
                workbook.close()
        except (BadZipFile, KeyError, OSError):
            raise ValueError("Unable to read XLSX workbook") from None
    else:
        raise ValueError("Supported holdings formats: .csv and .xlsx (save legacy .xls as .xlsx)")
    if len(rows) < 2 or len(rows) > MAX_ROWS + 1:
        raise ValueError(f"Supply a header and 1–{MAX_ROWS} holding rows")
    raw_headers = [_text(h).lower().replace(" ", "_") for h in rows[0]]
    headers = [ALIASES.get(h, h) for h in raw_headers]
    if len(set(headers)) != len(headers):
        raise ValueError("Duplicate column names")
    required = {"ticker", "market_value_usd"}
    if not required <= set(headers) or not {"asset_type", "asset_class"} & set(headers):
        raise ValueError("Required columns: ticker, market_value_usd, and asset_type or asset_class")
    grouped = defaultdict(list)
    for line, values in enumerate(rows[1:], start=2):
        if not any(_text(v) for v in values):
            continue
        if len(values) > len(headers):
            raise ValueError(f"Row {line}: more values than column headers")
        row = dict(zip(headers, values))
        if _text(row.get("currency")).upper() not in ("", "USD"):
            raise ValueError(f"Row {line}: only USD valuations are supported")
        group = _text(row.get("portfolio_id")) or _text(row.get("portfolio")) or "Uploaded portfolio"
        row["_line"] = line
        grouped[group].append(row)
    portfolios = []
    for index, (group, holdings_rows) in enumerate(grouped.items(), start=1):
        def consistent(key: str, default: str = "") -> str:
            values = {_text(r.get(key)) for r in holdings_rows}
            if len(values) > 1:
                raise ValueError(f"Portfolio {group}: inconsistent {key} values")
            return next(iter(values)) or default
        name = consistent("portfolio", group)
        as_of = consistent("as_of_date")
        if as_of:
            try:
                # Excel date cells arrive as datetime, with midnight time appended.
                as_of = date.fromisoformat(as_of.removesuffix(" 00:00:00")).isoformat()
            except ValueError:
                raise ValueError("as_of_date must be YYYY-MM-DD") from None
        total = sum(_number(r.get("market_value_usd"), f"Row {r['_line']} market_value_usd") for r in holdings_rows)
        if total <= 0 or not math.isfinite(total):
            raise ValueError(f"Portfolio {group}: total market value must be finite and positive")
        holdings = []
        for row in holdings_rows:
            value = _number(row.get("market_value_usd"), "market_value_usd")
            calculated_weight = value * 100 / total
            weight = row.get("weight_pct")
            if _text(weight) and abs(_number(weight, "weight_pct") - calculated_weight) > 0.1:
                raise ValueError(f"Row {row['_line']}: supplied weight disagrees with market value")
            holdings.append(Holding(
                ticker=_text(row.get("ticker")).upper(),
                asset_type=_text(row.get("asset_type")) or "Unspecified",
                weight_pct=calculated_weight, market_value_usd=value,
                asset_class_override=_text(row.get("asset_class")) or None,
                sector=_text(row.get("sector")) or None,
                sector_source=_text(row.get("sector_source")) or ("User supplied" if _text(row.get("sector")) else None),
            ))
        profile = profiles.get(group)
        age_text = consistent("age")
        age = _number(age_text, "age") if age_text else None
        if age is not None and (age != int(age) or age > 120):
            raise ValueError("age must be a whole number between 0 and 120")
        portfolio = Portfolio(
            portfolio_id=index, name=name, age=int(age) if age is not None else None,
            risk_tolerance=consistent("risk_tolerance", "Not supplied"),
            stated_value_usd=total, goal=consistent("goal", "Not supplied"),
            analysis_focus=consistent("analysis_focus"), holdings=tuple(holdings),
            as_of_date=as_of or None, input_source=f"Uploaded file: {Path(filename).name}",
        )
        if profile:
            from dataclasses import replace
            if abs(_number(profile[4], "profile value") - total) > max(1, total * 0.001):
                raise ValueError(f"Portfolio {group}: profile value disagrees with holdings total")
            portfolio = replace(portfolio, age=int(_number(profile[2], "age")),
                                risk_tolerance=_text(profile[3]), goal=_text(profile[5]),
                                analysis_focus=_text(profile[6]))
        PortfolioAnalysisAgent._validate_portfolio(portfolio)
        portfolios.append(portfolio)
    if not portfolios:
        raise ValueError("No holdings found")
    return tuple(portfolios)
