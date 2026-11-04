class KykeonError(Exception):
    status_code = 400
    code = "error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class PathRejected(KykeonError):
    code = "path_rejected"


class UnsupportedModel(KykeonError):
    status_code = 422
    code = "unsupported_model"


class MalformedCheckpoint(KykeonError):
    status_code = 422
    code = "malformed_checkpoint"


class ValidationFailed(KykeonError):
    status_code = 422
    code = "invalid_workflow"


class DeviceUnavailable(KykeonError):
    status_code = 409
    code = "device_unavailable"


class JobBusy(KykeonError):
    status_code = 409
    code = "job_busy"


class JobFailed(KykeonError):
    status_code = 422
    code = "job_failed"


class Forbidden(KykeonError):
    status_code = 403
    code = "forbidden"


class NotFound(KykeonError):
    status_code = 404
    code = "not_found"


class CancelledJob(Exception):
    """Raised inside the worker when the user cancels a job."""
