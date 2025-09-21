"""
Core module for trading bot infrastructure.

Contains configuration management, logging, event hub, and other foundational components.
"""

from .event_hub import EventHub, EventType, EventHubInterface

__all__ = ['EventHub', 'EventType', 'EventHubInterface']
