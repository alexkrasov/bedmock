"""Bedrock-compatible exceptions backed by botocore ClientError."""

from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError

from .logging_utils import safe_error_message


class BedmockError(ClientError):
    """Base class for Bedrock Runtime compatibility errors."""

    code = "BedmockError"
    status_code = 500

    def __init__(
        self,
        message: str,
        operation_name: str = "Bedmock",
        *,
        request_id: str | None = None,
        status_code: int | None = None,
        retry_attempts: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        metadata = {
            "RequestId": request_id or "bedmock-local",
            "HTTPStatusCode": status_code or self.status_code,
            "RetryAttempts": retry_attempts,
        }
        error_response: dict[str, Any] = {
            "Error": {
                "Code": self.code,
                "Message": safe_error_message(message),
            },
            "ResponseMetadata": metadata,
        }
        if extra:
            error_response.update(extra)
        super().__init__(error_response, operation_name)


class ValidationException(BedmockError):
    code = "ValidationException"
    status_code = 400


class AccessDeniedException(BedmockError):
    code = "AccessDeniedException"
    status_code = 403


class ThrottlingException(BedmockError):
    code = "ThrottlingException"
    status_code = 429


class ModelTimeoutException(BedmockError):
    code = "ModelTimeoutException"
    status_code = 504


class ServiceUnavailableException(BedmockError):
    code = "ServiceUnavailableException"
    status_code = 503


class InternalServerException(BedmockError):
    code = "InternalServerException"
    status_code = 500


class ResourceNotFoundException(BedmockError):
    code = "ResourceNotFoundException"
    status_code = 404


class ConflictException(BedmockError):
    code = "ConflictException"
    status_code = 409


class UnsupportedOperationException(BedmockError):
    code = "UnsupportedOperationException"
    status_code = 400


class ConfigurationError(ValidationException):
    code = "ValidationException"


class AmbiguousCodecError(ValidationException):
    code = "ValidationException"


class UnknownCodecError(ValidationException):
    code = "ValidationException"


class ExceptionsNamespace:
    ValidationException = ValidationException
    AccessDeniedException = AccessDeniedException
    ThrottlingException = ThrottlingException
    ModelTimeoutException = ModelTimeoutException
    ServiceUnavailableException = ServiceUnavailableException
    InternalServerException = InternalServerException
    ResourceNotFoundException = ResourceNotFoundException
    ConflictException = ConflictException
    UnsupportedOperationException = UnsupportedOperationException


HTTP_ERROR_MAP: dict[int, type[BedmockError]] = {
    400: ValidationException,
    401: AccessDeniedException,
    403: AccessDeniedException,
    404: ResourceNotFoundException,
    408: ModelTimeoutException,
    409: ConflictException,
    422: ValidationException,
    429: ThrottlingException,
    500: InternalServerException,
    502: ServiceUnavailableException,
    503: ServiceUnavailableException,
    504: ModelTimeoutException,
}


def error_from_http_status(status_code: int) -> type[BedmockError]:
    return HTTP_ERROR_MAP.get(status_code, InternalServerException)
