from intelligent_cc.decision import CaptionDecisionEngine
from intelligent_cc.models import AudioEvent, ReactionSignal


def test_accepts_event_when_audio_and_visual_scores_are_meaningful() -> None:
    suggestion = CaptionDecisionEngine().decide(
        AudioEvent("glass breaking", 0.62, 3.0, 4.0),
        ReactionSignal(0.55, 0.5, 0.6, 10),
        1,
    )

    assert suggestion is not None
    assert suggestion.text == "[glass breaking]"


def test_rejects_low_impact_event_without_reaction() -> None:
    suggestion = CaptionDecisionEngine().decide(
        AudioEvent("music", 0.28, 3.0, 4.0),
        ReactionSignal(0.05, 0.04, 0.0, 10),
        1,
    )

    assert suggestion is None
