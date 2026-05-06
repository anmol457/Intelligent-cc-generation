from intelligent_cc.models import CaptionSuggestion
from intelligent_cc.output import format_timestamp, write_srt


def test_format_timestamp_uses_srt_millisecond_format() -> None:
    assert format_timestamp(3723.456) == "01:02:03,456"


def test_write_srt(tmp_path) -> None:
    output = tmp_path / "captions.srt"
    write_srt(
        [
            CaptionSuggestion(
                index=1,
                label="honking",
                text="[honking]",
                start=1.0,
                end=2.25,
                audio_confidence=0.9,
                reaction_confidence=0.7,
                decision_score=0.8,
                reason="audio+visual",
            )
        ],
        output,
    )

    assert output.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:01,000 --> 00:00:02,250\n"
        "[honking]\n"
    )
