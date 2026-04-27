"""CORTEX exception hierarchy. All domain errors derive from CortexError."""


class CortexError(Exception):
    """Base class for all CORTEX domain exceptions."""


class CortexModelError(CortexError):
    """Raised when the model provider fails or returns an unexpected response."""


class CortexToolError(CortexError):
    """Raised when a tool fails to execute or violates its safety constraints."""


class CortexPlannerError(CortexError):
    """Raised when the planner cannot decompose a task into executable steps."""


class CortexMemoryError(CortexError):
    """Raised on failures in the memory subsystem."""
