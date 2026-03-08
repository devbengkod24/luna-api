"""Custom exceptions for the application."""


class VSAException(Exception):
    """Base exception for VSA API."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class BadRequestException(VSAException):
    """400 Bad Request."""

    def __init__(self, message: str = "Bad Request"):
        super().__init__(message, 400)


class UnauthorizedException(VSAException):
    """401 Unauthorized."""

    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message, 401)


class ForbiddenException(VSAException):
    """403 Forbidden."""

    def __init__(self, message: str = "Forbidden"):
        super().__init__(message, 403)


class NotFoundException(VSAException):
    """404 Not Found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, 404)


class ValidationException(VSAException):
    """422 Validation Error."""

    def __init__(self, message: str = "Validation Error"):
        super().__init__(message, 422)


class InternalServerException(VSAException):
    """500 Internal Server Error."""

    def __init__(self, message: str = "Internal Server Error"):
        super().__init__(message, 500)


class ServiceUnavailableException(VSAException):
    """503 Service Unavailable."""

    def __init__(self, message: str = "Service Unavailable"):
        super().__init__(message, 503)

