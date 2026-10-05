"""
Unit tests for Transcript data model.
"""

from datetime import datetime
from app.intelligence.transcript import Transcript


def test_transcript_properties():
    """Verify Transcript creation, empty check, and normalization."""
    t1 = Transcript(text="  hello world  ", is_final=False, confidence=0.95)
    assert not t1.is_empty
    assert t1.normalized_text == "hello world"
    assert t1.is_final is False
    assert t1.confidence == 0.95
    assert t1.transcript_id is not None
    assert isinstance(t1.timestamp, datetime)

    t2 = Transcript(text="   ", is_final=True)
    assert t2.is_empty
    assert t2.normalized_text == ""
    assert t2.is_final is True
