"""Custom exceptions for the credit scoring service."""


class CreditScoringError(Exception):
    """Base exception for all credit-scoring service errors."""


class MissingColumnsError(CreditScoringError):
    """Raised when required input columns are absent."""


class InvalidInputError(CreditScoringError):
    """Raised when input values fail type/range validation."""
