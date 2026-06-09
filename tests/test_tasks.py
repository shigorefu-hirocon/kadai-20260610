import math
import runpy
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.patches import Wedge


ROOT = Path(__file__).resolve().parents[1]
TEXT_COLUMNS = ["店舗", "カテゴリ", "商品名", "支払方法"]


def _prepare_workspace(tmp_path, *filenames):
    for filename in filenames:
        (tmp_path / filename).write_text((ROOT / filename).read_text(encoding="utf-8-sig"), encoding="utf-8")
    (tmp_path / "sales_dirty.csv").write_bytes((ROOT / "sales_dirty.csv").read_bytes())


def _expected_clean_sales():
    df = pd.read_csv(ROOT / "sales_dirty.csv", encoding="utf-8-sig")
    df = df.drop_duplicates().copy()

    for column in TEXT_COLUMNS:
        df[column] = df[column].astype("string").str.strip()

    df["単価"] = (
        df["単価"]
        .astype("string")
        .str.replace("¥", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["単価"] = pd.to_numeric(df["単価"], errors="coerce").astype(float)
    df["数量"] = pd.to_numeric(df["数量"], errors="coerce").astype(float)
    df["割引率"] = pd.to_numeric(df["割引率"], errors="coerce").astype(float)

    df["割引率"] = df["割引率"].fillna(0)
    df["数量"] = df["数量"].fillna(1)
    df["単価"] = df["単価"].fillna(df["単価"].median())

    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("不明")

    df["日付"] = pd.to_datetime(df["日付"], errors="coerce", format="mixed")
    df = df.dropna(subset=["日付"]).copy()
    df["売上金額"] = df["単価"] * df["数量"] * (1 - df["割引率"] / 100)
    df["年月"] = df["日付"].dt.to_period("M").astype(str)
    return df.reset_index(drop=True)


def _install_optional_import_stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "japanize_matplotlib", types.ModuleType("japanize_matplotlib"))
    monkeypatch.setitem(sys.modules, "setuptools", types.ModuleType("setuptools"))


def _run_script(tmp_path, monkeypatch, script_name):
    _install_optional_import_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    plt.close("all")
    runpy.run_path(str(tmp_path / script_name), run_name="__main__")


def _write_clean_fixture(tmp_path):
    expected = _expected_clean_sales()
    expected.to_csv(tmp_path / "sales_clean.csv", index=False)
    return expected


def _all_axes():
    return [axis for number in plt.get_fignums() for axis in plt.figure(number).axes]


def _bar_heights(axis):
    return [patch.get_height() for patch in axis.patches if patch.get_height() > 0]


def _bar_widths(axis):
    return [patch.get_width() for patch in axis.patches if patch.get_width() > 0]


def _assert_same_numbers(actual, expected, rel_tol=1e-6):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=1e-6)


def test_task01_creates_fully_clean_sales_csv(tmp_path, monkeypatch):
    _prepare_workspace(tmp_path, "task01_cleaning.py")

    _run_script(tmp_path, monkeypatch, "task01_cleaning.py")

    output = tmp_path / "sales_clean.csv"
    assert output.exists(), "task01_cleaning.py must save the cleaned data as sales_clean.csv"

    result = pd.read_csv(output)
    expected = _expected_clean_sales()

    assert len(result) == len(expected)
    assert result.duplicated().sum() == 0
    assert result.isna().sum().sum() == 0
    assert {"売上金額", "年月"}.issubset(result.columns)

    for column in TEXT_COLUMNS:
        assert not result[column].astype(str).str.startswith(" ").any()
        assert not result[column].astype(str).str.endswith(" ").any()
        assert "不明" in set(result[column])

    assert pd.api.types.is_numeric_dtype(result["単価"])
    assert pd.api.types.is_numeric_dtype(result["数量"])
    assert pd.api.types.is_numeric_dtype(result["割引率"])
    assert pd.to_datetime(result["日付"], errors="coerce").notna().all()
    assert result["年月"].astype(str).str.fullmatch(r"\d{4}-\d{2}").all()

    expected_sales = expected["単価"] * expected["数量"] * (1 - expected["割引率"] / 100)
    _assert_same_numbers(result["売上金額"].head(50), expected_sales.head(50))


def test_task02_plots_category_sales_in_descending_order(tmp_path, monkeypatch):
    _prepare_workspace(tmp_path, "task02.py")
    expected = _write_clean_fixture(tmp_path).groupby("カテゴリ")["売上金額"].sum().sort_values(ascending=False)

    _run_script(tmp_path, monkeypatch, "task02.py")

    matching_axes = [axis for axis in _all_axes() if len(_bar_heights(axis)) == len(expected)]
    assert matching_axes, "task02.py must draw one bar for each category"
    _assert_same_numbers(_bar_heights(matching_axes[0]), expected.tolist())


def test_task03_plots_store_sales_as_horizontal_bars(tmp_path, monkeypatch):
    _prepare_workspace(tmp_path, "task03.py")
    expected = _write_clean_fixture(tmp_path).groupby("店舗")["売上金額"].sum().sort_values(ascending=False)

    _run_script(tmp_path, monkeypatch, "task03.py")

    matching_axes = [axis for axis in _all_axes() if len(_bar_widths(axis)) == len(expected)]
    assert matching_axes, "task03.py must draw one horizontal bar for each store"
    _assert_same_numbers(_bar_widths(matching_axes[0]), expected.tolist())


def test_task04_plots_monthly_sales_line(tmp_path, monkeypatch):
    _prepare_workspace(tmp_path, "task04.py")
    expected = _write_clean_fixture(tmp_path).groupby("年月")["売上金額"].sum().sort_index()

    _run_script(tmp_path, monkeypatch, "task04.py")

    lines = [line for axis in _all_axes() for line in axis.lines if len(line.get_ydata()) == len(expected)]
    assert lines, "task04.py must draw a line for monthly sales"
    _assert_same_numbers(lines[0].get_ydata(), expected.tolist())


def test_task05_plots_payment_method_pie_chart(tmp_path, monkeypatch):
    _prepare_workspace(tmp_path, "task05.py")
    expected = _write_clean_fixture(tmp_path)["支払方法"].value_counts()

    _run_script(tmp_path, monkeypatch, "task05.py")

    wedges = [patch for axis in _all_axes() for patch in axis.patches if isinstance(patch, Wedge)]
    assert len(wedges) == len(expected), "task05.py must draw one pie slice for each payment method"
    actual_angles = sorted(round(wedge.theta2 - wedge.theta1, 2) for wedge in wedges)
    expected_angles = sorted(round(count / expected.sum() * 360, 2) for count in expected)
    _assert_same_numbers(actual_angles, expected_angles, rel_tol=1e-3)


def test_task06_plots_price_histogram_with_labels(tmp_path, monkeypatch):
    _prepare_workspace(tmp_path, "task06.py")
    _write_clean_fixture(tmp_path)

    _run_script(tmp_path, monkeypatch, "task06.py")

    axes = _all_axes()
    assert any(axis.patches for axis in axes), "task06.py must draw a histogram"
    assert any(axis.get_title() for axis in axes), "task06.py must set a graph title"
    assert any(axis.get_xlabel() for axis in axes), "task06.py must set the x-axis label"
    assert any(axis.get_ylabel() for axis in axes), "task06.py must set the y-axis label"


def test_task07_plots_three_required_scatter_charts(tmp_path, monkeypatch):
    _prepare_workspace(tmp_path, "task07.py")
    expected = _write_clean_fixture(tmp_path)

    _run_script(tmp_path, monkeypatch, "task07.py")

    collections = [
        collection
        for axis in _all_axes()
        for collection in axis.collections
        if isinstance(collection, PathCollection) and len(collection.get_offsets()) == len(expected)
    ]
    assert len(collections) >= 3, "task07.py must draw three scatter plots"

    expected_pairs = [
        ("割引率", "売上金額"),
        ("割引率", "数量"),
        ("単価", "売上金額"),
    ]
    for collection, (x_column, y_column) in zip(collections[:3], expected_pairs):
        offsets = collection.get_offsets()
        _assert_same_numbers(offsets[:50, 0], expected[x_column].head(50))
        _assert_same_numbers(offsets[:50, 1], expected[y_column].head(50))
