import unittest

from mood_analysis import detect_local_mood


class LocalMoodAnalysisTests(unittest.TestCase):
    def test_detects_common_indonesian_moods(self) -> None:
        cases = [
            ("Hari ini aku senang dan bersyukur.", "happy"),
            ("Aku merasa tenang dan lega.", "calm"),
            ("Aku sedih dan kesepian.", "sad"),
            ("Aku cemas banget dan takut.", "anxious"),
        ]

        for transcript, expected_state in cases:
            result = detect_local_mood(transcript, "low")
            self.assertEqual(result.state, expected_state)
            self.assertGreater(result.confidence, 0.5)

    def test_high_risk_always_resolves_to_distressed(self) -> None:
        result = detect_local_mood("Aku baik-baik saja", "high")
        self.assertEqual(result.state, "distressed")
        self.assertEqual(result.confidence, 0.95)

    def test_unknown_text_is_neutral(self) -> None:
        result = detect_local_mood("Aku pergi ke kampus hari ini", "low")
        self.assertEqual(result.state, "neutral")
        self.assertEqual(result.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
