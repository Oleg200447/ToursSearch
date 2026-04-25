class ServiceError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ValidationServiceError(ServiceError):
    def __init__(self, message: str):
        super().__init__(message=message, status_code=422)


class DataAccessError(ServiceError):
    def __init__(self, message: str = "Database operation failed"):
        super().__init__(message=message, status_code=503)


class ExternalDependencyError(ServiceError):
    def __init__(self, message: str = "External dependency failed"):
        super().__init__(message=message, status_code=502)
