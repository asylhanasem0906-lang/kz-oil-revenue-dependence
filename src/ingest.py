"""Сборка набора данных из data/raw в data/processed.

Запуск из корня проекта:  python src/ingest.py
Ничего не вводится вручную: каждый шаг читает файл из data/raw.
"""
from pathlib import Path

import openpyxl
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "data" / "processed"

FIRST_YEAR, LAST_YEAR = 2007, 2025


def load_fiscal() -> pd.DataFrame:
    """Нефтяные налоги Нацфонда, доходы и трансферты госбюджета (тыс. тенге)."""
    df = pd.read_csv(RAW / "raw_manual_input.csv", encoding="utf-8-sig")
    df = df[(df["year"] >= FIRST_YEAR) & (df["year"] <= LAST_YEAR)].copy()
    cols = ["oil_tax_nf_thousand_kzt", "state_income_thousand_kzt", "state_transfers_thousand_kzt"]
    df = df[["year"] + cols].astype(float)
    df["year"] = df["year"].astype(int)
    return df


def load_brent() -> pd.DataFrame:
    """FRED DCOILBRENTEU, среднегодовая цена, USD за баррель."""
    df = pd.read_csv(RAW / "brent_usd_per_bbl_annual_avg.csv")
    df["year"] = pd.to_datetime(df["observation_date"]).dt.year
    return df.rename(columns={"DCOILBRENTEU": "brent_usd_per_bbl"})[["year", "brent_usd_per_bbl"]]


def load_fx() -> pd.DataFrame:
    """Официальный среднегодовой курс USD/KZT (Нацбанк), по таблице, составленной из файлов usd_kzt_annual_avg_*."""
    df = pd.read_csv(RAW / "usd_kzt_annual_avg_input.csv", encoding="utf-8-sig")
    return df[["year", "usd_kzt_annual_avg"]]


def load_oil() -> pd.DataFrame:
    """Добыча нефти (БНС, таблица 1991-2025), тыс. тонн -> млн тонн. Строки ищутся по названию, не по номеру."""
    wb = openpyxl.load_workbook(RAW / "extra_material_oil_output.xlsx", data_only=True, read_only=True)
    rows = list(wb["горнодоб"].iter_rows(values_only=True))
    header = next(r for r in rows if sum(str(v).strip().isdigit() for v in r if v is not None) > 10)
    year_col = {int(v): i for i, v in enumerate(header) if str(v).strip().isdigit()}

    def national_row(label_start: str):
        hits = [i for i, r in enumerate(rows)
                if r[0] is not None and str(r[0]).strip().lower().startswith(label_start)]
        assert len(hits) == 1, f"ожидалась одна строка '{label_start}', найдено {len(hits)}"
        row = rows[hits[0] + 1]
        assert str(row[0]).strip().lower().startswith("республика казахстан"), label_start
        return {y: float(row[c]) / 1000 for y, c in year_col.items() if FIRST_YEAR <= y <= LAST_YEAR}

    combined = national_row("нефть, включая конденсат")
    crude = national_row("нефть сырая")
    cond = national_row("конденсат газовый")
    return pd.DataFrame({
        "year": list(combined),
        "oil_crude_mln_t": [crude[y] for y in combined],
        "condensate_mln_t": [cond[y] for y in combined],
        "oil_incl_condensate_mln_t": [combined[y] for y in combined],
    })


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    oil, inc, tr = (df["oil_tax_nf_thousand_kzt"], df["state_income_thousand_kzt"],
                    df["state_transfers_thousand_kzt"])
    # Доля нефтяных налогов в доходах консолидированного бюджета (без двойного счёта трансфертов)
    df["Y_share_pct"] = oil / (inc - tr + oil) * 100
    # Отношение нефтяных налогов Нацфонда к доходам госбюджета (не доля)
    df["Y_ratio_pct"] = oil / inc * 100
    return df


def check(df: pd.DataFrame) -> None:
    assert df["year"].tolist() == list(range(FIRST_YEAR, LAST_YEAR + 1)), "в данных пропущены годы"
    assert not df.isna().any().any(), "в данных есть пропуски"
    assert (df["state_income_thousand_kzt"] > df["state_transfers_thousand_kzt"]).all()
    assert ((df["oil_crude_mln_t"] + df["condensate_mln_t"] - df["oil_incl_condensate_mln_t"]).abs() < 1e-3).all()
    assert df["Y_share_pct"].between(0, 100).all()


def main() -> None:
    df = load_fiscal()
    for part in (load_brent(), load_fx(), load_oil()):
        df = df.merge(part, on="year", how="left")
    df = add_indicators(df).sort_values("year").reset_index(drop=True)
    check(df)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "master_dataset.csv", index=False, encoding="utf-8")
    print(f"Сохранено: {OUT / 'master_dataset.csv'} ({len(df)} строк)")


if __name__ == "__main__":
    main()
