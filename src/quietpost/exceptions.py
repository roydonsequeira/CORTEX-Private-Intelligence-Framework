"""Quietpost exception hierarchy. All domain errors derive from QuietpostError."""


class QuietpostError(Exception):
    """Base class for all Quietpost domain exceptions."""


class QuietpostModelError(QuietpostError):
    """Raised when the model provider fails or returns an unexpected response."""


class QuietpostToolError(QuietpostError):
    """Raised when a tool fails to execute or violates its safety constraints."""


class QuietpostPlannerError(QuietpostError):
    """Raised when the planner cannot decompose a task into executable steps."""


class QuietpostMemoryError(QuietpostError):
    """Raised on failures in the memory subsystem."""


class QuietpostConfigError(QuietpostError):
    """Raised when the configuration would break one of Quietpost's rules."""
