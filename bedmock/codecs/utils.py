"""Shared codec conversion helpers."""

from __future__ import annotations

import base64
import json
from typing import Any

from bedmock.canonical import (
    CanonicalCachePointBlock,
    CanonicalContentBlock,
    CanonicalImageBlock,
    CanonicalJsonBlock,
    CanonicalMessage,
    CanonicalReasoningBlock,
    CanonicalResponseFormat,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
    CanonicalToolResultBlock,
    CanonicalToolUseBlock,
    CanonicalUsage,
)
from bedmock.exceptions import ValidationException

ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}


def require_type(value: Any, expected: type, field: str) -> Any:
    if not isinstance(value, expected):
        raise ValidationException(f"{field} must be {expected.__name__}")
    return value


def read_text(blocks: list[CanonicalContentBlock]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, CanonicalTextBlock):
            parts.append(block.text)
        elif isinstance(block, CanonicalJsonBlock):
            parts.append(json.dumps(block.value, ensure_ascii=False))
    return "".join(parts)


def cache_point_from_converse(value: Any) -> CanonicalCachePointBlock:
    cache_point = require_type(value, dict, "cachePoint")
    cache_type = cache_point.get("type")
    if cache_type != "default":
        raise ValidationException("cachePoint.type must be 'default'")
    ttl = cache_point.get("ttl")
    if ttl is not None and ttl not in {"5m", "1h"}:
        raise ValidationException("cachePoint.ttl must be '5m' or '1h'")
    return CanonicalCachePointBlock(type="default", ttl=ttl)


def validate_base64_image(media_type: str, data_base64: str, *, max_bytes: int) -> None:
    if media_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise ValidationException(f"Unsupported image MIME type: {media_type}")
    try:
        decoded = base64.b64decode(data_base64, validate=True)
    except Exception as exc:
        raise ValidationException("Image data must be valid base64") from exc
    if len(decoded) > max_bytes:
        raise ValidationException("Image data exceeds BEDMOCK_MAX_IMAGE_BYTES")


def anthropic_content_to_blocks(content: Any) -> list[CanonicalContentBlock]:
    if isinstance(content, str):
        return [CanonicalTextBlock(content)]
    if not isinstance(content, list):
        raise ValidationException("Anthropic message content must be string or list")

    blocks: list[CanonicalContentBlock] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValidationException("Anthropic content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            blocks.append(CanonicalTextBlock(str(block.get("text", ""))))
        elif block_type == "image":
            source = require_type(block.get("source"), dict, "image.source")
            if source.get("type") != "base64":
                raise ValidationException("Only Anthropic base64 images are supported")
            media_type = str(source.get("media_type") or "")
            data = str(source.get("data") or "")
            validate_base64_image(media_type, data, max_bytes=10 * 1024 * 1024)
            blocks.append(CanonicalImageBlock(media_type=media_type, data_base64=data))
        elif block_type == "tool_use":
            tool_id = str(block.get("id") or "")
            if not tool_id:
                raise ValidationException("tool_use.id is required")
            blocks.append(
                CanonicalToolUseBlock(
                    id=tool_id,
                    name=str(block.get("name") or ""),
                    arguments=block.get("input") or {},
                )
            )
        elif block_type == "tool_result":
            blocks.append(
                CanonicalToolResultBlock(
                    tool_use_id=str(block.get("tool_use_id") or ""),
                    content=anthropic_content_to_blocks(block.get("content") or []),
                    is_error=bool(block.get("is_error", False)),
                )
            )
        else:
            raise ValidationException(f"Unsupported Anthropic content block type: {block_type}")
    return blocks


def anthropic_blocks_from_system(system: Any) -> list[CanonicalContentBlock]:
    if system is None:
        return []
    if isinstance(system, str):
        return [CanonicalTextBlock(system)]
    return anthropic_content_to_blocks(system)


def anthropic_blocks_from_canonical(blocks: list[CanonicalContentBlock]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, CanonicalTextBlock):
            output.append({"type": "text", "text": block.text})
        elif isinstance(block, CanonicalImageBlock):
            if block.data_base64:
                output.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.media_type,
                            "data": block.data_base64,
                        },
                    }
                )
            elif block.url:
                output.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": block.url, "media_type": block.media_type},
                    }
                )
        elif isinstance(block, CanonicalToolUseBlock):
            arguments = block.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"arguments": arguments}
            output.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": arguments}
            )
        elif isinstance(block, CanonicalToolResultBlock):
            output.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": anthropic_blocks_from_canonical(block.content),
                    "is_error": block.is_error,
                }
            )
        elif isinstance(block, CanonicalJsonBlock):
            output.append({"type": "text", "text": json.dumps(block.value, ensure_ascii=False)})
        elif isinstance(block, CanonicalReasoningBlock):
            output.append({"type": "text", "text": block.text})
    return output


def converse_content_to_blocks(content: Any) -> list[CanonicalContentBlock]:
    if not isinstance(content, list):
        raise ValidationException("Converse message content must be a list")
    blocks: list[CanonicalContentBlock] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValidationException("Converse content blocks must be objects")
        if "text" in block:
            blocks.append(CanonicalTextBlock(str(block["text"])))
        elif "cachePoint" in block:
            blocks.append(cache_point_from_converse(block["cachePoint"]))
        elif "json" in block:
            blocks.append(CanonicalJsonBlock(block["json"]))
        elif "image" in block:
            image = require_type(block["image"], dict, "image")
            source = require_type(image.get("source"), dict, "image.source")
            media_format = str(image.get("format") or "")
            media_type = f"image/{media_format}" if "/" not in media_format else media_format
            if "bytes" in source:
                raw = source["bytes"]
                data = base64.b64encode(raw).decode("ascii") if isinstance(raw, bytes) else str(raw)
                validate_base64_image(media_type, data, max_bytes=10 * 1024 * 1024)
                blocks.append(CanonicalImageBlock(media_type=media_type, data_base64=data))
            elif "url" in source:
                blocks.append(CanonicalImageBlock(media_type=media_type, url=str(source["url"])))
            else:
                raise ValidationException("Converse image source must include bytes or url")
        elif "toolUse" in block:
            tool_use = require_type(block["toolUse"], dict, "toolUse")
            blocks.append(
                CanonicalToolUseBlock(
                    id=str(tool_use.get("toolUseId") or ""),
                    name=str(tool_use.get("name") or ""),
                    arguments=tool_use.get("input") or {},
                )
            )
        elif "toolResult" in block:
            result = require_type(block["toolResult"], dict, "toolResult")
            blocks.append(
                CanonicalToolResultBlock(
                    tool_use_id=str(result.get("toolUseId") or ""),
                    content=converse_content_to_blocks(result.get("content") or []),
                    is_error=result.get("status") == "error",
                )
            )
        elif "reasoningContent" in block:
            reasoning = block["reasoningContent"]
            if isinstance(reasoning, dict):
                text = reasoning.get("text") or reasoning.get("reasoningText", {}).get("text") or ""
            else:
                text = str(reasoning)
            blocks.append(CanonicalReasoningBlock(text=text, redacted=text == "[REDACTED]"))
        else:
            raise ValidationException(f"Unsupported Converse content block: {sorted(block)}")
    return blocks


def converse_blocks_from_canonical(blocks: list[CanonicalContentBlock]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, CanonicalTextBlock):
            output.append({"text": block.text})
        elif isinstance(block, CanonicalCachePointBlock):
            cache_point: dict[str, str] = {"type": block.type}
            if block.ttl is not None:
                cache_point["ttl"] = block.ttl
            output.append({"cachePoint": cache_point})
        elif isinstance(block, CanonicalJsonBlock):
            output.append({"json": block.value})
        elif isinstance(block, CanonicalReasoningBlock):
            output.append({"reasoningContent": {"text": block.text}})
        elif isinstance(block, CanonicalImageBlock):
            media_format = block.media_type.split("/", 1)[-1]
            if block.data_base64:
                output.append(
                    {"image": {"format": media_format, "source": {"bytes": block.data_base64}}}
                )
            elif block.url:
                output.append({"image": {"format": media_format, "source": {"url": block.url}}})
        elif isinstance(block, CanonicalToolUseBlock):
            arguments = block.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"arguments": arguments}
            output.append(
                {
                    "toolUse": {
                        "toolUseId": block.id,
                        "name": block.name,
                        "input": arguments,
                    }
                }
            )
        elif isinstance(block, CanonicalToolResultBlock):
            output.append(
                {
                    "toolResult": {
                        "toolUseId": block.tool_use_id,
                        "content": converse_blocks_from_canonical(block.content),
                        "status": "error" if block.is_error else "success",
                    }
                }
            )
    return output


def messages_from_anthropic(messages: Any) -> list[CanonicalMessage]:
    require_type(messages, list, "messages")
    result: list[CanonicalMessage] = []
    for message in messages:
        require_type(message, dict, "messages[]")
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ValidationException("Anthropic messages only support user and assistant roles")
        result.append(
            CanonicalMessage(role=role, content=anthropic_content_to_blocks(message["content"]))
        )
    return result


def messages_from_converse(messages: Any) -> list[CanonicalMessage]:
    require_type(messages, list, "messages")
    result: list[CanonicalMessage] = []
    for message in messages:
        require_type(message, dict, "messages[]")
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ValidationException("Converse messages only support user and assistant roles")
        result.append(
            CanonicalMessage(role=role, content=converse_content_to_blocks(message["content"]))
        )
    return result


def tools_from_anthropic(tools: Any) -> list[CanonicalTool]:
    if tools in (None, []):
        return []
    require_type(tools, list, "tools")
    result: list[CanonicalTool] = []
    for tool in tools:
        require_type(tool, dict, "tools[]")
        result.append(
            CanonicalTool(
                name=str(tool.get("name") or ""),
                description=tool.get("description"),
                input_schema=dict(tool.get("input_schema") or {}),
                metadata={"source": "anthropic"},
            )
        )
    return result


def tools_from_converse(tool_config: Any) -> list[CanonicalTool]:
    if not tool_config:
        return []
    require_type(tool_config, dict, "toolConfig")
    tools = require_type(tool_config.get("tools") or [], list, "toolConfig.tools")
    result: list[CanonicalTool] = []
    for item in tools:
        require_type(item, dict, "toolConfig.tools[]")
        spec = item.get("toolSpec")
        if not isinstance(spec, dict):
            raise ValidationException("Only toolSpec Converse tools are supported")
        input_schema = spec.get("inputSchema", {}).get("json", {})
        result.append(
            CanonicalTool(
                name=str(spec.get("name") or ""),
                description=spec.get("description"),
                input_schema=dict(input_schema),
                metadata={"source": "converse"},
            )
        )
    return result


def tool_choice_from_anthropic(choice: Any) -> CanonicalToolChoice | None:
    if not choice:
        return None
    if isinstance(choice, str):
        if choice == "auto":
            return CanonicalToolChoice("auto")
        if choice == "none":
            return CanonicalToolChoice("none")
    require_type(choice, dict, "tool_choice")
    choice_type = choice.get("type")
    if choice_type == "auto":
        return CanonicalToolChoice("auto")
    if choice_type == "none":
        return CanonicalToolChoice("none")
    if choice_type in {"any", "required"}:
        return CanonicalToolChoice("required")
    if choice_type == "tool":
        return CanonicalToolChoice("specific", tool_name=str(choice.get("name") or ""))
    raise ValidationException(f"Unsupported Anthropic tool_choice: {choice_type}")


def tool_choice_from_converse(tool_config: Any) -> CanonicalToolChoice | None:
    if not tool_config:
        return None
    require_type(tool_config, dict, "toolConfig")
    choice = tool_config.get("toolChoice")
    if not choice:
        return None
    require_type(choice, dict, "toolConfig.toolChoice")
    if "auto" in choice:
        return CanonicalToolChoice("auto")
    if "any" in choice:
        return CanonicalToolChoice("required")
    if "tool" in choice:
        tool = require_type(choice["tool"], dict, "toolConfig.toolChoice.tool")
        return CanonicalToolChoice("specific", tool_name=str(tool.get("name") or ""))
    raise ValidationException(f"Unsupported Converse toolChoice: {sorted(choice)}")


def response_format_from_extensions(fields: dict[str, Any]) -> CanonicalResponseFormat | None:
    response_format = fields.get("response_format") or fields.get("responseFormat")
    if not response_format:
        return None
    require_type(response_format, dict, "response_format")
    fmt_type = response_format.get("type")
    if fmt_type in {"json_object", "json"}:
        return CanonicalResponseFormat("json_object", metadata={"source": response_format})
    if fmt_type in {"json_schema", "schema"}:
        schema = response_format.get("json_schema") or response_format.get("schema") or {}
        name = schema.get("name") if isinstance(schema, dict) else None
        strict = schema.get("strict") if isinstance(schema, dict) else None
        payload_schema = schema.get("schema") if isinstance(schema, dict) else schema
        return CanonicalResponseFormat(
            "json_schema",
            schema=payload_schema,
            name=name,
            strict=strict,
            metadata={"source": response_format},
        )
    raise ValidationException(f"Unsupported response_format type: {fmt_type}")


def finish_reason_from_provider(reason: str | None) -> str | None:
    mapping = {
        "stop": "end_turn",
        "end_turn": "end_turn",
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "stop_sequence": "stop_sequence",
        "tool_calls": "tool_use",
        "tool_use": "tool_use",
        "content_filter": "content_filtered",
        "content_filtered": "content_filtered",
    }
    if reason is None:
        return None
    return mapping.get(reason, "unknown")


def finish_reason_to_openai(reason: str | None) -> str | None:
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "content_filtered": "content_filter",
        "error": "stop",
        "unknown": "stop",
    }
    if reason is None:
        return None
    return mapping.get(reason, "stop")


def usage_dict(usage: CanonicalUsage) -> dict[str, int]:
    result: dict[str, int] = {}
    if usage.input_tokens is not None:
        result["input_tokens"] = usage.input_tokens
    if usage.output_tokens is not None:
        result["output_tokens"] = usage.output_tokens
    if usage.total_tokens is not None:
        result["total_tokens"] = usage.total_tokens
    return result
