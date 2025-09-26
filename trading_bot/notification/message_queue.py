"""
Persistent message queue for failed Discord webhook deliveries.

Provides reliable message queuing with persistence, retry scheduling, and
graceful degradation when Discord webhooks fail. Follows SOLID principles
with dependency injection for flexible storage backends.
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


class MessageStatus(Enum):
    """Status of queued messages."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class MessagePriority(Enum):
    """Priority levels for messages."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class QueuedMessage:
    """
    Represents a message in the queue.

    Attributes:
        id: Unique message identifier
        content: Message payload as dictionary
        priority: Message priority level
        status: Current message status
        created_at: Message creation timestamp
        scheduled_at: When message should be processed
        attempts: Number of processing attempts
        max_attempts: Maximum allowed attempts
        last_error: Last processing error message
        webhook_url: Target Discord webhook URL
        retry_delays: List of delay periods for retries
        metadata: Additional message metadata
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    created_at: float = field(default_factory=time.time)
    scheduled_at: float = field(default_factory=time.time)
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    webhook_url: str = ""
    retry_delays: List[float] = field(default_factory=lambda: [1.0, 5.0, 15.0])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for storage."""
        data = asdict(self)
        data["priority"] = self.priority.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueuedMessage":
        """Create message from dictionary."""
        # Handle enum conversions
        if "priority" in data and isinstance(data["priority"], int):
            data["priority"] = MessagePriority(data["priority"])
        if "status" in data and isinstance(data["status"], str):
            data["status"] = MessageStatus(data["status"])

        return cls(**data)

    def should_retry(self) -> bool:
        """Check if message should be retried."""
        return (
            self.status in [MessageStatus.PENDING, MessageStatus.FAILED]
            and self.attempts < self.max_attempts
            and time.time() >= self.scheduled_at
        )

    def calculate_next_retry(self) -> float:
        """Calculate next retry time based on attempts and delays."""
        if self.attempts < len(self.retry_delays):
            delay = self.retry_delays[self.attempts]
        else:
            # Use exponential backoff for additional attempts
            delay = self.retry_delays[-1] * (
                2 ** (self.attempts - len(self.retry_delays))
            )

        return time.time() + delay


@dataclass
class QueueConfig:
    """
    Configuration for message queue behavior.

    Attributes:
        max_size: Maximum queue size (0 for unlimited)
        retention_hours: Hours to keep completed/failed messages
        batch_size: Number of messages to process in batch
        processing_interval: Seconds between processing cycles
        persistence_enabled: Whether to persist messages to storage
        auto_cleanup_enabled: Whether to automatically clean old messages
        cleanup_interval: Seconds between cleanup cycles
        storage_path: Path to storage file/directory
    """

    max_size: int = 1000
    retention_hours: int = 24
    batch_size: int = 10
    processing_interval: float = 1.0
    persistence_enabled: bool = True
    auto_cleanup_enabled: bool = True
    cleanup_interval: float = 3600.0  # 1 hour
    storage_path: str = "discord_message_queue.db"


class IMessageStorage(ABC):
    """Interface for message storage backends."""

    @abstractmethod
    def save_message(self, message: QueuedMessage) -> None:
        """Save message to storage."""

    @abstractmethod
    def load_messages(self) -> List[QueuedMessage]:
        """Load all messages from storage."""

    @abstractmethod
    def update_message(self, message: QueuedMessage) -> None:
        """Update existing message in storage."""

    @abstractmethod
    def delete_message(self, message_id: str) -> None:
        """Delete message from storage."""

    @abstractmethod
    def cleanup_old_messages(self, retention_hours: int) -> int:
        """Remove old messages. Returns count of deleted messages."""

    @abstractmethod
    def close(self) -> None:
        """Close storage connection."""


class SqliteMessageStorage(IMessageStorage):
    """SQLite-based message storage implementation."""

    def __init__(self, db_path: str) -> None:
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    scheduled_at REAL NOT NULL,
                    attempts INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    last_error TEXT,
                    webhook_url TEXT NOT NULL,
                    retry_delays TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """
            )

            # Create indexes for performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status_scheduled
                ON messages(status, scheduled_at)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON messages(created_at)
            """
            )

    def save_message(self, message: QueuedMessage) -> None:
        """Save message to SQLite database."""
        with self._lock:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO messages (
                            id, content, priority, status, created_at, scheduled_at,
                            attempts, max_attempts, last_error, webhook_url,
                            retry_delays, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            message.id,
                            json.dumps(message.content),
                            message.priority.value,
                            message.status.value,
                            message.created_at,
                            message.scheduled_at,
                            message.attempts,
                            message.max_attempts,
                            message.last_error,
                            message.webhook_url,
                            json.dumps(message.retry_delays),
                            json.dumps(message.metadata),
                        ),
                    )

            except sqlite3.Error as e:
                self._logger.error(f"Failed to save message to database: {e}")
                raise

    def load_messages(self) -> List[QueuedMessage]:
        """Load all messages from database."""
        with self._lock:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT id, content, priority, status, created_at, scheduled_at,
                               attempts, max_attempts, last_error, webhook_url,
                               retry_delays, metadata
                        FROM messages
                        ORDER BY priority DESC, scheduled_at ASC
                    """
                    )

                    messages = []
                    for row in cursor.fetchall():
                        message = QueuedMessage(
                            id=row[0],
                            content=json.loads(row[1]),
                            priority=MessagePriority(row[2]),
                            status=MessageStatus(row[3]),
                            created_at=row[4],
                            scheduled_at=row[5],
                            attempts=row[6],
                            max_attempts=row[7],
                            last_error=row[8],
                            webhook_url=row[9],
                            retry_delays=json.loads(row[10]),
                            metadata=json.loads(row[11]),
                        )
                        messages.append(message)

                    return messages

            except sqlite3.Error as e:
                self._logger.error(f"Failed to load messages from database: {e}")
                return []

    def update_message(self, message: QueuedMessage) -> None:
        """Update existing message."""
        self.save_message(message)  # INSERT OR REPLACE handles updates

    def delete_message(self, message_id: str) -> None:
        """Delete message from database."""
        with self._lock:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))

            except sqlite3.Error as e:
                self._logger.error(f"Failed to delete message from database: {e}")
                raise

    def cleanup_old_messages(self, retention_hours: int) -> int:
        """Remove old completed/failed messages."""
        cutoff_time = time.time() - (retention_hours * 3600)

        with self._lock:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    cursor = conn.execute(
                        """
                        DELETE FROM messages
                        WHERE created_at < ?
                        AND status IN (?, ?, ?)
                    """,
                        (
                            cutoff_time,
                            MessageStatus.COMPLETED.value,
                            MessageStatus.FAILED.value,
                            MessageStatus.EXPIRED.value,
                        ),
                    )

                    return cursor.rowcount

            except sqlite3.Error as e:
                self._logger.error(f"Failed to cleanup old messages: {e}")
                return 0

    def close(self) -> None:
        """
        Close database connection (no persistent connection in this implementation).
        """


class InMemoryMessageStorage(IMessageStorage):
    """In-memory message storage for testing or temporary use."""

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._messages: Dict[str, QueuedMessage] = {}
        self._lock = threading.Lock()

    def save_message(self, message: QueuedMessage) -> None:
        """Save message to memory."""
        with self._lock:
            self._messages[message.id] = message

    def load_messages(self) -> List[QueuedMessage]:
        """Load all messages from memory."""
        with self._lock:
            return list(self._messages.values())

    def update_message(self, message: QueuedMessage) -> None:
        """Update existing message."""
        self.save_message(message)

    def delete_message(self, message_id: str) -> None:
        """Delete message from memory."""
        with self._lock:
            self._messages.pop(message_id, None)

    def cleanup_old_messages(self, retention_hours: int) -> int:
        """Remove old messages from memory."""
        cutoff_time = time.time() - (retention_hours * 3600)
        removed_count = 0

        with self._lock:
            messages_to_remove = []
            for msg_id, message in self._messages.items():
                if message.created_at < cutoff_time and message.status in [
                    MessageStatus.COMPLETED,
                    MessageStatus.FAILED,
                    MessageStatus.EXPIRED,
                ]:
                    messages_to_remove.append(msg_id)

            for msg_id in messages_to_remove:
                del self._messages[msg_id]
                removed_count += 1

            return removed_count

    def close(self) -> None:
        """Clear memory storage."""
        with self._lock:
            self._messages.clear()


class MessageQueue:
    """
    Persistent message queue for Discord webhook failures.

    Provides reliable message queuing with configurable retry policies,
    persistence, and automatic cleanup. Thread-safe for concurrent use.
    """

    def __init__(
        self, config: QueueConfig, storage: Optional[IMessageStorage] = None
    ) -> None:
        """
        Initialize message queue.

        Args:
            config: Queue configuration
            storage: Optional storage backend
        """
        self._config = config
        self._storage = storage or SqliteMessageStorage(config.storage_path)
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._logger = logging.getLogger(__name__)

        # Load existing messages from storage
        if config.persistence_enabled:
            self._load_persisted_messages()

    def _load_persisted_messages(self) -> None:
        """Load persisted messages from storage."""
        try:
            messages = self._storage.load_messages()
            self._logger.info(f"Loaded {len(messages)} persisted messages from storage")

            # Reset processing messages to pending on startup
            for message in messages:
                if message.status == MessageStatus.PROCESSING:
                    message.status = MessageStatus.PENDING
                    self._storage.update_message(message)

        except Exception as e:
            self._logger.error(f"Failed to load persisted messages: {e}")

    async def enqueue(
        self,
        content: Dict[str, Any],
        webhook_url: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        max_attempts: int = 3,
        retry_delays: Optional[List[float]] = None,
    ) -> str:
        """
        Add message to queue.

        Args:
            content: Message content as dictionary
            webhook_url: Target webhook URL
            priority: Message priority
            max_attempts: Maximum retry attempts
            retry_delays: Custom retry delays

        Returns:
            str: Message ID

        Raises:
            ValueError: If queue is full
        """
        # Check queue size limit
        if self._config.max_size > 0:
            current_messages = self._storage.load_messages()
            active_count = sum(
                1
                for msg in current_messages
                if msg.status
                not in [
                    MessageStatus.COMPLETED,
                    MessageStatus.FAILED,
                    MessageStatus.EXPIRED,
                ]
            )

            if active_count >= self._config.max_size:
                raise ValueError(f"Queue is full (max size: {self._config.max_size})")

        # Create message
        message = QueuedMessage(
            content=content,
            webhook_url=webhook_url,
            priority=priority,
            max_attempts=max_attempts,
            retry_delays=retry_delays or [1.0, 5.0, 15.0],
        )

        # Save to storage if persistence enabled
        if self._config.persistence_enabled:
            self._storage.save_message(message)

        self._logger.debug(
            f"Enqueued message {message.id} with priority {priority.name}"
        )
        return message.id

    def get_pending_messages(self, limit: Optional[int] = None) -> List[QueuedMessage]:
        """
        Get pending messages ready for processing.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List[QueuedMessage]: Messages ready for processing
        """
        current_time = time.time()
        messages = self._storage.load_messages()

        # Filter for processable messages
        pending_messages = [
            msg
            for msg in messages
            if msg.should_retry() and msg.scheduled_at <= current_time
        ]

        # Sort by priority and scheduled time
        pending_messages.sort(key=lambda x: (-x.priority.value, x.scheduled_at))

        if limit:
            pending_messages = pending_messages[:limit]

        return pending_messages

    def update_message_status(
        self, message_id: str, status: MessageStatus, error: Optional[str] = None
    ) -> bool:
        """
        Update message status.

        Args:
            message_id: Message identifier
            status: New status
            error: Optional error message

        Returns:
            bool: True if message was updated
        """
        try:
            messages = self._storage.load_messages()
            message = next((msg for msg in messages if msg.id == message_id), None)

            if not message:
                self._logger.warning(
                    f"Message {message_id} not found for status update"
                )
                return False

            message.status = status
            if error:
                message.last_error = error

            # Schedule retry if failed and attempts remaining
            if (
                status == MessageStatus.FAILED
                and message.attempts < message.max_attempts
            ):
                message.scheduled_at = message.calculate_next_retry()
                message.status = MessageStatus.PENDING

            self._storage.update_message(message)

            self._logger.debug(f"Updated message {message_id} status to {status.value}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to update message status: {e}")
            return False

    def mark_message_processing(self, message_id: str) -> bool:
        """Mark message as currently being processed."""
        return self.update_message_status(message_id, MessageStatus.PROCESSING)

    def mark_message_completed(self, message_id: str) -> bool:
        """Mark message as successfully completed."""
        return self.update_message_status(message_id, MessageStatus.COMPLETED)

    def mark_message_failed(self, message_id: str, error: str) -> bool:
        """Mark message as failed with error details."""
        return self.update_message_status(message_id, MessageStatus.FAILED, error)

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        messages = self._storage.load_messages()

        stats = {
            "total_messages": len(messages),
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "expired": 0,
            "by_priority": {"urgent": 0, "high": 0, "normal": 0, "low": 0},
            "oldest_pending": None,
            "newest_message": None,
        }

        for message in messages:
            # Count by status
            stats[message.status.value] += 1

            # Count by priority
            priority_name = message.priority.name.lower()
            stats["by_priority"][priority_name] += 1

            # Track timestamps
            if message.status == MessageStatus.PENDING:
                if (
                    stats["oldest_pending"] is None
                    or message.scheduled_at < stats["oldest_pending"]
                ):
                    stats["oldest_pending"] = message.scheduled_at

            if (
                stats["newest_message"] is None
                or message.created_at > stats["newest_message"]
            ):
                stats["newest_message"] = message.created_at

        return stats

    async def cleanup_old_messages(self) -> int:
        """Clean up old completed/failed messages."""
        if not self._config.auto_cleanup_enabled:
            return 0

        removed_count = self._storage.cleanup_old_messages(self._config.retention_hours)

        if removed_count > 0:
            self._logger.info(f"Cleaned up {removed_count} old messages")

        return removed_count

    def start_background_processing(self) -> None:
        """Start background message processing."""
        if self._running:
            self._logger.warning("Background processing already running")
            return

        self._running = True

        # Start processor task
        loop = asyncio.get_event_loop()
        self._processor_task = loop.create_task(self._background_processor())

        # Start cleanup task if enabled
        if self._config.auto_cleanup_enabled:
            self._cleanup_task = loop.create_task(self._background_cleanup())

        self._logger.info("Started background message processing")

    async def stop_background_processing(self) -> None:
        """Stop background message processing."""
        self._running = False

        # Cancel tasks
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._logger.info("Stopped background message processing")

    async def _background_processor(self) -> None:
        """Background task to process pending messages."""
        while self._running:
            try:
                # This would be implemented with actual message processing logic
                # For now, it's a placeholder for the processing cycle
                await asyncio.sleep(self._config.processing_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in background processor: {e}")
                await asyncio.sleep(5.0)  # Back off on errors

    async def _background_cleanup(self) -> None:
        """Background task to clean up old messages."""
        while self._running:
            try:
                await self.cleanup_old_messages()
                await asyncio.sleep(self._config.cleanup_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in background cleanup: {e}")
                await asyncio.sleep(60.0)  # Back off on errors

    def close(self) -> None:
        """Close queue and storage."""
        if self._storage:
            self._storage.close()


def create_message_queue(
    storage_path: Optional[str] = None, max_size: int = 1000, persistence: bool = True
) -> MessageQueue:
    """
    Factory function to create message queue.

    Args:
        storage_path: Path to storage file
        max_size: Maximum queue size
        persistence: Whether to enable persistence

    Returns:
        MessageQueue: Configured message queue
    """
    config = QueueConfig(
        max_size=max_size,
        persistence_enabled=persistence,
        storage_path=storage_path or "discord_message_queue.db",
    )

    if persistence:
        storage = SqliteMessageStorage(config.storage_path)
    else:
        storage = InMemoryMessageStorage()

    return MessageQueue(config, storage)
