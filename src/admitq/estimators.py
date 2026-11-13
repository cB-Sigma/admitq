"""Pre-admission cost estimation.

An estimator turns a parsed request body into a :class:`Cost` (prompt tokens plus the
output tokens the request *may* generate).  The estimate is used to reserve capacity and
budget; it is corrected with real usage when the response finishes.

The default :class:`CharEstimator` is a character-count heuristic with no dependencies.
:class:`TiktokenEstimator` and :class:`TokenizersEstimator` use real tokenizers when the
optional extras are installed.  Anything with ``estimate`` / ``estimate_raw`` /
``chars_per_token`` satisfies :class:`CostEstimator`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "BodyEstimator",
    "CharEstimator",
    "Cost",
    "CostEstimator",
    "TiktokenEstimator",
    "TokenizersEstimator",
]

#: Endpoint kinds that generate no output tokens.
_NO_OUTPUT = frozenset({"embeddings"})


@dataclass(frozen=True, slots=True)
class Cost:
    """Token cost of one request."""

    prompt_tokens: int
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")


@runtime_checkable
class CostEstimator(Protocol):
    """Interface for cost estimators."""

    #: Used to convert streamed text back to tokens when the engine reports no usage.
    chars_per_token: float

    def estimate(self, body: Mapping[str, Any], endpoint: str) -> Cost:
        """Estimate the cost of a parsed JSON request body."""
        ...

    def estimate_raw(self, nbytes: int, endpoint: str) -> Cost:
        """Estimate the cost of a body that could not be parsed (non-JSON or too large)."""
        ...


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


class BodyEstimator:
    """Walks OpenAI-style request bodies and counts tokens with a pluggable text counter.

    Args:
        count_text: ``str -> int`` token counter.
        chars_per_token: characters per token, used for unparsed bodies and for streamed
            text when the engine reports no usage.
        default_output_tokens: output tokens assumed when the request sets no
            ``max_tokens`` / ``max_completion_tokens`` / ``max_output_tokens``.
        output_factor: fraction of the requested maximum output to reserve.  ``1.0`` is the
            worst case.  Lower it if your clients routinely set a large ``max_tokens`` and
            use little of it; the ``admitq_estimate_ratio`` histogram tells you by how much.
        max_output_tokens: optional cap on the reserved output (for example the model's
            context size).
        tokens_per_message: fixed overhead per chat message (role, separators).
        tokens_per_image: flat cost of an image/audio/file content part.
    """

    def __init__(
        self,
        count_text: Callable[[str], int],
        *,
        chars_per_token: float = 4.0,
        default_output_tokens: int = 512,
        output_factor: float = 1.0,
        max_output_tokens: int | None = None,
        tokens_per_message: int = 4,
        tokens_per_image: int = 512,
    ) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        if not 0.0 < output_factor <= 1.0:
            raise ValueError("output_factor must be in (0, 1]")
        if default_output_tokens < 0:
            raise ValueError("default_output_tokens must be non-negative")
        self._count = count_text
        self.chars_per_token = float(chars_per_token)
        self.default_output_tokens = default_output_tokens
        self.output_factor = output_factor
        self.max_output_tokens = max_output_tokens
        self.tokens_per_message = tokens_per_message
        self.tokens_per_image = tokens_per_image

    # -- public API -------------------------------------------------------------------

    def estimate(self, body: Mapping[str, Any], endpoint: str) -> Cost:
        prompt = 0
        sequences = 1

        messages = body.get("messages")
        if isinstance(messages, list):
            prompt += self._count_messages(messages)

        raw_prompt = body.get("prompt")
        if raw_prompt is not None:
            tokens, count = self._count_input(raw_prompt)
            prompt += tokens
            sequences = max(sequences, count)

        raw_input = body.get("input")
        if raw_input is not None:
            if endpoint == "responses" and isinstance(raw_input, list):
                prompt += self._count_messages(raw_input)
            else:
                prompt += self._count_input(raw_input)[0]

        for key in ("instructions", "system", "suffix"):
            value = body.get(key)
            if isinstance(value, str):
                prompt += self._count(value)

        for key in ("tools", "functions", "response_format", "text"):
            value = body.get(key)
            if isinstance(value, list | dict) and value:
                prompt += self._count(json.dumps(value, separators=(",", ":")))

        output = 0 if endpoint in _NO_OUTPUT else self._output_tokens(body) * sequences
        return Cost(prompt_tokens=prompt, output_tokens=output)

    def estimate_raw(self, nbytes: int, endpoint: str) -> Cost:
        prompt = math.ceil(max(0, nbytes) / self.chars_per_token)
        output = 0 if endpoint in _NO_OUTPUT else self._scale_output(self.default_output_tokens)
        return Cost(prompt_tokens=prompt, output_tokens=output)

    # -- helpers ----------------------------------------------------------------------

    def _scale_output(self, requested: int) -> int:
        if self.max_output_tokens is not None:
            requested = min(requested, self.max_output_tokens)
        return max(0, math.ceil(requested * self.output_factor))

    def _output_tokens(self, body: Mapping[str, Any]) -> int:
        requested: int | None = None
        for key in ("max_completion_tokens", "max_tokens", "max_output_tokens"):
            value = _as_int(body.get(key))
            if value is not None and value >= 0:
                requested = value
                break
        if requested is None:
            # llama.cpp / Ollama native spellings.
            for key in ("n_predict", "num_predict"):
                value = _as_int(body.get(key))
                if value is not None and value > 0:
                    requested = value
                    break
        if requested is None:
            options = body.get("options")
            if isinstance(options, Mapping):
                value = _as_int(options.get("num_predict"))
                if value is not None and value > 0:
                    requested = value
        if requested is None:
            requested = self.default_output_tokens
        choices = 1
        for key in ("n", "best_of"):
            value = _as_int(body.get(key))
            if value is not None and value > choices:
                choices = value
        return self._scale_output(requested) * choices

    def _count_messages(self, messages: list[Any]) -> int:
        total = 0
        for message in messages:
            if isinstance(message, str):
                total += self._count(message)
                continue
            if not isinstance(message, Mapping):
                continue
            total += self.tokens_per_message
            total += self._count_content(message.get("content"))
            for key in ("name", "arguments", "output", "reasoning_content"):
                value = message.get(key)
                if isinstance(value, str):
                    total += self._count(value)
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                total += self._count(json.dumps(tool_calls, separators=(",", ":")))
        return total

    def _count_content(self, content: Any) -> int:
        if isinstance(content, str):
            return self._count(content)
        if not isinstance(content, list):
            return 0
        total = 0
        for part in content:
            if isinstance(part, str):
                total += self._count(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    total += self._count(text)
                elif part.get("type") not in (None, "text", "input_text", "output_text"):
                    total += self.tokens_per_image
        return total

    def _count_input(self, value: Any) -> tuple[int, int]:
        """Count a ``prompt``/``input`` field.  Returns ``(tokens, number_of_sequences)``."""
        if isinstance(value, str):
            return self._count(value), 1
        if isinstance(value, list):
            if not value:
                return 0, 1
            if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
                return len(value), 1  # one pre-tokenized sequence
            total = 0
            for item in value:
                if isinstance(item, str):
                    total += self._count(item)
                elif isinstance(item, list):
                    total += len(item)  # pre-tokenized
                elif isinstance(item, Mapping):
                    total += self._count_messages([item])
            return total, len(value)
        return 0, 1


class CharEstimator(BodyEstimator):
    """Dependency-free heuristic: ``ceil(len(text) / chars_per_token)``.

    Four characters per token is a common rule of thumb for English with BPE tokenizers.
    It under-counts dense scripts (CJK text is closer to one token per character) and
    code; lower ``chars_per_token`` for those workloads or use a real tokenizer.
    """

    def __init__(self, *, chars_per_token: float = 4.0, **kwargs: Any) -> None:
        cpt = float(chars_per_token)
        if cpt <= 0:
            raise ValueError("chars_per_token must be positive")

        def count(text: str) -> int:
            return math.ceil(len(text) / cpt)

        super().__init__(count, chars_per_token=cpt, **kwargs)


class TiktokenEstimator(BodyEstimator):
    """Counts with a ``tiktoken`` encoding (``pip install admitq[tiktoken]``).

    Exact only if the served model actually uses that encoding; for open-weight models it
    is still a heuristic, usually a closer one than characters.  Tokenization runs on the
    event loop: fine for typical prompts, measurable for very large ones.
    """

    def __init__(self, encoding: str = "cl100k_base", **kwargs: Any) -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "TiktokenEstimator needs tiktoken: pip install 'admitq[tiktoken]'"
            ) from exc
        enc = tiktoken.get_encoding(encoding)

        def count(text: str) -> int:
            return len(enc.encode(text, disallowed_special=()))

        super().__init__(count, **kwargs)


class TokenizersEstimator(BodyEstimator):
    """Counts with a Hugging Face ``tokenizers.Tokenizer`` (``pip install admitq[tokenizers]``).

    Pass a loaded tokenizer, or a path to a ``tokenizer.json`` file.  Using the tokenizer
    of the model you serve gives exact prompt counts up to chat-template overhead.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any) -> None:
        if isinstance(tokenizer, str):
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "TokenizersEstimator needs tokenizers: pip install 'admitq[tokenizers]'"
                ) from exc
            tokenizer = Tokenizer.from_file(tokenizer)
        tok = tokenizer

        def count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False).ids)

        super().__init__(count, **kwargs)
