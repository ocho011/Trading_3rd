"""
Core module for trading bot infrastructure.

Contains configuration, logging, event hub, and foundational components.
"""

from .event_hub import EventHub, EventHubInterface, EventType

__all__ = ["EventHub", "EventType", "EventHubInterface"]
