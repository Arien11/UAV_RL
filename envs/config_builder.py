import os
from typing import Any

import yaml


class Configuration:
    """A class to handle configuration data with attribute-style access."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize a Configuration object with nested attribute access.

        Args:
            **kwargs: Key-value pairs to be set as attributes.
        """
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Configuration(**value))
            elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                setattr(self, key, [Configuration(**item) for item in value])
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        """Return a string representation of the configuration."""
        return str(self.__dict__)

    def __getattr__(self, name: str) -> None:
        """Return None for non-existent attributes instead of raising an error."""
        return None

    def to_dict(self):
        """Convert the Configuration object back to a dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Configuration):
                result[key] = value.to_dict()
            elif isinstance(value, list) and value and isinstance(value[0], Configuration):
                result[key] = [item.to_dict() if isinstance(item, Configuration) else item for item in value]
            else:
                result[key] = value
        return result