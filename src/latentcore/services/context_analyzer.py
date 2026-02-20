from __future__ import annotations

import re
from dataclasses import dataclass, field

from latentcore.config import Settings
from latentcore.models.openai_compat import ChatMessage
from latentcore.utils.token_utils import count_tokens

# Pattern to detect existing VQ latent references in message content
VQ_REF_PATTERN = re.compile(r"\[VQ_LATENT_REF_([a-fA-F0-9]+)\]")


@dataclass
class CompressSegment:
    """A message segment identified for compression."""

    text: str
    original_index: int
    token_count: int
    role: str


@dataclass
class AnalysisResult:
    """Result of context analysis on a messages array."""

    messages_to_keep: list[ChatMessage] = field(default_factory=list)
    segments_to_compress: list[CompressSegment] = field(default_factory=list)
    existing_latent_refs: list[str] = field(default_factory=list)
    total_input_tokens: int = 0
    compressible_tokens: int = 0


class ContextAnalyzer:
    """Analyzes a messages array and partitions it into keep-as-text vs compress segments.

    Decision rules:
    1. System messages are always kept as-is.
    2. The most recent N message pairs (default 2) are always kept.
    3. Messages with existing VQ_LATENT_REF placeholders are kept (already compressed).
    4. All other messages exceeding the token threshold are marked for compression.
    """

    def __init__(self, settings: Settings):
        self.threshold = settings.compress_threshold_tokens
        self.keep_recent_pairs = 2  # Keep last N user/assistant turn pairs

    def analyze(self, messages: list[ChatMessage]) -> AnalysisResult:
        result = AnalysisResult()

        if not messages:
            return result

        # Separate system messages from conversation messages
        system_msgs = []
        conversation_msgs = []
        for i, msg in enumerate(messages):
            if msg.role == "system":
                system_msgs.append((i, msg))
            else:
                conversation_msgs.append((i, msg))

        # System messages always kept
        for _, msg in system_msgs:
            result.messages_to_keep.append(msg)
            if msg.content:
                refs = VQ_REF_PATTERN.findall(str(msg.content))
                result.existing_latent_refs.extend(refs)

        # Determine which conversation messages to protect (most recent turns)
        protected_count = self.keep_recent_pairs * 2  # 2 messages per turn pair
        protected_indices = set()
        if conversation_msgs:
            recent = conversation_msgs[-protected_count:]
            for idx, _ in recent:
                protected_indices.add(idx)

        # Analyze each conversation message
        for original_idx, msg in conversation_msgs:
            content = str(msg.content) if msg.content else ""

            # Check for existing latent refs
            refs = VQ_REF_PATTERN.findall(content)
            if refs:
                result.existing_latent_refs.extend(refs)
                result.messages_to_keep.append(msg)
                continue

            token_count = count_tokens(content) if content else 0
            result.total_input_tokens += token_count

            # Protected recent messages always kept
            if original_idx in protected_indices:
                result.messages_to_keep.append(msg)
                continue

            # Compress if exceeds threshold
            if token_count > self.threshold:
                result.segments_to_compress.append(
                    CompressSegment(
                        text=content,
                        original_index=original_idx,
                        token_count=token_count,
                        role=msg.role,
                    )
                )
                result.compressible_tokens += token_count
            else:
                result.messages_to_keep.append(msg)

        return result
