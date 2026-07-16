import unittest
from unittest.mock import patch

import main


class FailingCommunicate:
    async def save(self, path: str) -> None:
        raise RuntimeError("tts unavailable")


class VoiceProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_tts_failure_returns_text_only_response(self) -> None:
        processor = main.VoiceProcessor()
        config = main.VoiceRequestConfig(
            language="en",
            tts_voice="",
            tts_rate="-10%",
            enabled=True,
            preferred_name="",
            context="",
        )

        with patch.object(
            main.edge_tts, "Communicate", return_value=FailingCommunicate()
        ):
            audio_url = await processor.generate_audio("I am here with you.", config)

        self.assertIsNone(audio_url)


if __name__ == "__main__":
    unittest.main()
