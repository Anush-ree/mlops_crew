"""Focused tests for Phase 2 monitoring and evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlops_crew.data import TEXT_COLUMN
from mlops_crew.evaluation.metrics import binary_classification_report
from mlops_crew.monitoring.divergence import _js_distance
from mlops_crew.monitoring.inference_latency import measure_latency
from mlops_crew.monitoring.resource_monitor import ResourceMonitor


def test_binary_classification_report_tracks_f2_and_false_negative_rate() -> None:
    """Model selection should expose recall-weighted metrics for phishing risk."""
    report = binary_classification_report(
        y_true=[1, 1, 1, 0, 0],
        y_pred=[1, 0, 1, 1, 0],
        positive_label=1,
    )

    assert report["true_positive"] == 2
    assert report["false_negative"] == 1
    assert report["false_positive"] == 1
    assert report["f2"] == pytest.approx(2 / 3)
    assert report["false_negative_rate"] == pytest.approx(1 / 3)


def test_js_distance_is_zero_for_matching_distributions() -> None:
    """Divergence should not report drift when categorical distributions match."""
    left = {"0": 70, "1": 30}
    right = {"0": 140, "1": 60}

    assert _js_distance(left, right) == pytest.approx(0.0)


def test_resource_monitor_stops_cleanly_and_writes_csv(tmp_path: Path) -> None:
    """Background sampler should stop before CSV export and be restartable."""
    monitor = ResourceMonitor(interval_seconds=0.05)
    monitor.start()
    monitor.stop()
    output = tmp_path / "usage.csv"
    monitor.write_csv(output)

    assert output.exists()
    frame = pd.read_csv(output)
    assert not frame.empty

    monitor.start()
    monitor.stop()


def test_measure_latency_reports_each_batch_and_repeat() -> None:
    """Latency monitoring should produce stable rows for configured batches."""

    class ConstantModel:
        def predict(self, texts: pd.Series) -> list[int]:
            return [0] * len(texts)

    data = pd.DataFrame({TEXT_COLUMN: ["a", "b", "c", "d"]})

    report = measure_latency(ConstantModel(), data,
                             batch_sizes=[1, 3, 10], repeats=2)

    assert report["batch_size"].tolist() == [1, 1, 3, 3, 4, 4]
    assert report["repeat"].tolist() == [1, 2, 1, 2, 1, 2]
    assert (report["elapsed_seconds"] >= 0).all()
    assert (report["milliseconds_per_record"] >= 0).all()
