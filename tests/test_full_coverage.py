import json
import runpy
import sys

import pytest

from src.aiops_pipeline import load_data, run_pipeline
from src.anomaly_detector import AnomalyDetector
from src.calculations import area_of_circle, get_nth_fibonacci
from src.event_producer import EventProducer
from src.event_topic import EventTopic


def test_calculations_error_and_loop_paths():
    with pytest.raises(ValueError, match="Radius cannot be negative"):
        area_of_circle(-1)

    with pytest.raises(ValueError, match="n cannot be negative"):
        get_nth_fibonacci(-1)

    assert get_nth_fibonacci(10) == 55


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("response_time_ms", 501, "High response time"),
        ("cpu_percent", 81, "High CPU utilization"),
        ("memory_percent", 81, "High memory utilization"),
    ],
)
def test_detector_reports_each_threshold_reason(field, value, reason):
    record = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "test-service",
        "response_time_ms": 100,
        "cpu_percent": 20,
        "memory_percent": 20,
        "log_level": "INFO",
        "message": "test",
    }
    record[field] = value

    event = AnomalyDetector().detect(record)

    assert event["reasons"] == [reason]
    assert event["source"] == record


def test_detector_reports_warning_and_multiple_reasons():
    record = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "test-service",
        "response_time_ms": 501,
        "cpu_percent": 81,
        "memory_percent": 81,
        "log_level": "WARNING",
        "message": "test",
    }

    event = AnomalyDetector().detect(record)

    assert event["reasons"] == [
        "High response time",
        "High CPU utilization",
        "High memory utilization",
        "Error log detected",
    ]


def test_event_topic_copy_and_clear():
    topic = EventTopic("test-events")
    event = {"type": "ANOMALY"}

    topic.publish(event)
    messages = topic.get_messages()
    messages.clear()

    assert topic.get_messages() == [event]
    topic.clear()
    assert topic.get_messages() == []


def test_empty_event_is_not_published():
    producer = EventProducer(EventTopic("test-events"))

    assert producer.publish(None) is False


def test_load_data_and_pipeline_with_anomaly(tmp_path):
    records = [
        {
            "timestamp": "2026-09-20T10:00:00",
            "service": "test-service",
            "response_time_ms": 501,
            "cpu_percent": 20,
            "memory_percent": 20,
            "log_level": "INFO",
            "message": "slow",
        }
    ]
    data_file = tmp_path / "service_data.json"
    data_file.write_text(json.dumps(records), encoding="utf-8")

    assert load_data(data_file) == records
    result = run_pipeline(data_file)

    assert result["records_processed"] == 1
    assert len(result["anomalies_detected"]) == 1
    assert result["events_consumed"] == []


def test_pipeline_with_no_anomalies(tmp_path):
    record = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "test-service",
        "response_time_ms": 100,
        "cpu_percent": 20,
        "memory_percent": 20,
        "log_level": "INFO",
        "message": "healthy",
    }
    data_file = tmp_path / "healthy.json"
    data_file.write_text(json.dumps([record]), encoding="utf-8")

    result = run_pipeline(data_file)

    assert result["records_processed"] == 1
    assert result["anomalies_detected"] == []
    assert result["events_consumed"] == []


def test_pipeline_script_entry_point(capsys, monkeypatch):
    monkeypatch.chdir("/workspaces/github-skills-challenge")
    monkeypatch.syspath_prepend("/workspaces/github-skills-challenge/src")
    from event_consumer import EventConsumer as ScriptEventConsumer

    event = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "test-service",
        "type": "ANOMALY",
        "reasons": ["High response time"],
    }
    monkeypatch.setattr(ScriptEventConsumer, "consume", lambda self: [event])

    runpy.run_path("src/aiops_pipeline.py", run_name="__main__")

    output = capsys.readouterr().out
    assert "AIOps Pipeline Result" in output
    assert "Records processed:" in output