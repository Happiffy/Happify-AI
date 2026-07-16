from dataclasses import dataclass


@dataclass(frozen=True)
class LocalMoodResult:
    state: str
    confidence: float
    risk_level: str
    requires_referral: bool


MOOD_TERMS = {
    "happy": ("senang", "bahagia", "gembira", "bersyukur", "happy", "glad", "joy"),
    "calm": ("tenang", "lega", "damai", "calm", "relaxed", "relax"),
    "sad": ("sedih", "murung", "kecewa", "kesepian", "sendiri", "sad", "lonely"),
    "anxious": (
        "cemas",
        "khawatir",
        "takut",
        "panik",
        "tegang",
        "anxious",
        "worried",
        "panic",
    ),
}


def detect_local_mood(content: str, risk_level: str) -> LocalMoodResult:
    if risk_level in {"high", "crisis"}:
        return LocalMoodResult(
            state="distressed",
            confidence=0.95,
            risk_level=risk_level,
            requires_referral=True,
        )

    text = content.lower()
    matches = [
        (state, sum(1 for term in terms if term in text))
        for state, terms in MOOD_TERMS.items()
    ]
    state, score = max(matches, key=lambda item: item[1])
    if score == 0:
        state = "neutral"
        confidence = 0.5
    else:
        confidence = min(0.9, 0.6 + score * 0.1)

    return LocalMoodResult(
        state=state,
        confidence=confidence,
        risk_level=risk_level,
        requires_referral=False,
    )
