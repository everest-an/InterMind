from __future__ import annotations

import tiktoken

_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    if encoding_name not in _ENCODING_CACHE:
        _ENCODING_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _ENCODING_CACHE[encoding_name]


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    enc = _get_encoding(encoding_name)
    return len(enc.encode(text))


def count_messages_tokens(messages: list[dict], encoding_name: str = "cl100k_base") -> int:
    """Estimate token count for a list of chat messages.

    Follows OpenAI's counting convention: ~4 tokens overhead per message.
    """
    enc = _get_encoding(encoding_name)
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        for key, value in msg.items():
            if isinstance(value, str):
                total += len(enc.encode(value))
    total += 3  # reply priming
    return total


def truncate_to_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    enc = _get_encoding(encoding_name)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
