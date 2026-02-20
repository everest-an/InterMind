import pytest

from latentcore.config import Settings
from latentcore.models.openai_compat import ChatMessage
from latentcore.services.context_analyzer import ContextAnalyzer


@pytest.fixture
def analyzer():
    settings = Settings(
        upstream_base_url="http://test:8080/v1",
        compress_threshold_tokens=50,
    )
    return ContextAnalyzer(settings)


class TestContextAnalyzer:
    def test_empty_messages(self, analyzer):
        result = analyzer.analyze([])
        assert len(result.messages_to_keep) == 0
        assert len(result.segments_to_compress) == 0

    def test_system_message_always_kept(self, analyzer):
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant. " * 100),
            ChatMessage(role="user", content="Hi"),
        ]
        result = analyzer.analyze(messages)
        assert any(m.role == "system" for m in result.messages_to_keep)
        assert len(result.segments_to_compress) == 0

    def test_recent_messages_protected(self, analyzer):
        messages = [
            ChatMessage(role="user", content="old message " * 200),
            ChatMessage(role="assistant", content="old reply " * 200),
            ChatMessage(role="user", content="recent question"),
            ChatMessage(role="assistant", content="recent answer"),
        ]
        result = analyzer.analyze(messages)

        # Last 2 pairs (4 messages) should be protected, but we only have 4 total
        # The old messages with >50 tokens should be marked for compression
        kept_contents = [str(m.content) for m in result.messages_to_keep]
        assert "recent question" in kept_contents
        assert "recent answer" in kept_contents

    def test_long_old_messages_compressed(self, analyzer):
        # Create a message that definitely exceeds 50 tokens
        long_text = "This is a very long message with lots of content. " * 20
        messages = [
            ChatMessage(role="user", content=long_text),
            ChatMessage(role="assistant", content="Long reply " * 50),
            ChatMessage(role="user", content="New question"),
            ChatMessage(role="assistant", content="New answer"),
            ChatMessage(role="user", content="Latest"),
            ChatMessage(role="assistant", content="Latest reply"),
        ]
        result = analyzer.analyze(messages)
        assert len(result.segments_to_compress) > 0
        assert result.compressible_tokens > 0

    def test_short_messages_kept(self, analyzer):
        messages = [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ]
        result = analyzer.analyze(messages)
        assert len(result.segments_to_compress) == 0
        assert len(result.messages_to_keep) == 2

    def test_existing_vq_refs_detected(self, analyzer):
        messages = [
            ChatMessage(
                role="user",
                content="Context: [VQ_LATENT_REF_abc123] Please continue.",
            ),
        ]
        result = analyzer.analyze(messages)
        assert "abc123" in result.existing_latent_refs
