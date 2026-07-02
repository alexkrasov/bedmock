"""Codec registry."""

from __future__ import annotations

from typing import Any

from .amazon_nova import AmazonNovaCodec
from .amazon_titan_text import AmazonTitanTextCodec
from .anthropic_legacy import AnthropicLegacyCodec
from .anthropic_messages import AnthropicMessagesCodec
from .base import BedrockModelCodec
from .generic_prompt import GenericPromptCodec
from .meta_llama import MetaLlamaCodec
from .mistral import MistralCodec


def built_in_codecs() -> list[BedrockModelCodec]:
    return [
        AnthropicMessagesCodec(),
        AmazonNovaCodec(),
        AnthropicLegacyCodec(),
        MetaLlamaCodec(),
        MistralCodec(),
        AmazonTitanTextCodec(),
        GenericPromptCodec(),
    ]


class CodecRegistry:
    def __init__(self, codecs: list[BedrockModelCodec] | None = None) -> None:
        self._codecs = codecs or built_in_codecs()
        ids: set[str] = set()
        for codec in self._codecs:
            if codec.codec_id in ids:
                raise ValueError(f"Duplicate codec registered: {codec.codec_id}")
            ids.add(codec.codec_id)

    @property
    def codecs(self) -> list[BedrockModelCodec]:
        return list(self._codecs)

    def names(self) -> list[str]:
        return [codec.codec_id for codec in self._codecs]

    def get(self, codec_id: str) -> BedrockModelCodec:
        for codec in self._codecs:
            if codec.codec_id == codec_id:
                return codec
        raise KeyError(codec_id)

    def detect(
        self, model_id: str, body: dict[str, Any], source_codec: str | None = None
    ) -> BedrockModelCodec:
        if source_codec:
            return self.get(source_codec)
        matches = [codec for codec in self._codecs if codec.can_decode(model_id, body)]
        if not matches:
            from bedmock.exceptions import UnknownCodecError

            raise UnknownCodecError(
                "No Bedrock model-family codec matched "
                f"modelId={model_id!r}, fields={sorted(body)}. See docs/adding-a-codec.md. "
                f"Registered codecs: {', '.join(self.names())}"
            )
        matches.sort(key=lambda codec: codec.priority)
        best_priority = matches[0].priority
        best = [codec for codec in matches if codec.priority == best_priority]
        if len(best) > 1:
            from bedmock.exceptions import AmbiguousCodecError

            raise AmbiguousCodecError(
                "Multiple Bedrock codecs matched "
                f"modelId={model_id!r}: {', '.join(codec.codec_id for codec in best)}"
            )
        return best[0]


DEFAULT_CODEC_REGISTRY = CodecRegistry()
