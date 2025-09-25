"""
Unit tests for message queue implementation.

Tests persistent message storage, retry scheduling, priority handling,
and cleanup mechanisms for Discord webhook message queue system.
"""

import asyncio
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from trading_bot.notification.message_queue import (
    InMemoryMessageStorage, MessagePriority, MessageQueue, MessageQueueManager,
    QueueConfig, QueuedMessage, SqliteMessageStorage,
    create_message_queue_manager)


class TestQueuedMessage:
    """Test cases for QueuedMessage data class."""

    def test_message_creation(self):
        """Test creating a queued message."""
        message = QueuedMessage(
            content="Test message",
            webhook_url="https://discord.com/api/webhooks/test",
            priority=MessagePriority.HIGH,
        )

        assert message.content == "Test message"
        assert message.webhook_url == "https://discord.com/api/webhooks/test"
        assert message.priority == MessagePriority.HIGH
        assert message.retry_count == 0
        assert message.max_retries == 3
        assert message.next_retry is None
        assert message.created_at is not None

    def test_message_serialization(self):
        """Test message to/from dict conversion."""
        original = QueuedMessage(
            content="Test",
            webhook_url="https://discord.com/test",
            priority=MessagePriority.NORMAL,
            retry_count=2,
        )

        # Convert to dict
        message_dict = original.to_dict()
        assert message_dict["content"] == "Test"
        assert message_dict["priority"] == "normal"
        assert message_dict["retry_count"] == 2

        # Convert back from dict
        restored = QueuedMessage.from_dict(message_dict)
        assert restored.content == original.content
        assert restored.webhook_url == original.webhook_url
        assert restored.priority == original.priority
        assert restored.retry_count == original.retry_count

    def test_message_retry_increment(self):
        """Test incrementing retry count and scheduling."""
        message = QueuedMessage(content="Test", webhook_url="https://discord.com/test")

        # Calculate next retry time (exponential backoff)
        delay_seconds = 2 ** (message.retry_count + 1)  # 2 seconds for first retry
        next_retry = datetime.utcnow() + timedelta(seconds=delay_seconds)

        message.increment_retry(delay_seconds)

        assert message.retry_count == 1
        assert message.next_retry is not None
        # Allow small time difference due to execution time
        assert abs((message.next_retry - next_retry).total_seconds()) < 1

    def test_message_can_retry(self):
        """Test retry eligibility logic."""
        message = QueuedMessage(
            content="Test", webhook_url="https://discord.com/test", max_retries=2
        )

        # Should be able to retry initially
        assert message.can_retry()

        # After max retries, should not be able to retry
        message.retry_count = 2
        assert not message.can_retry()

    def test_message_is_ready_for_retry(self):
        """Test retry timing logic."""
        message = QueuedMessage(content="Test", webhook_url="https://discord.com/test")

        # Initially ready (no next_retry set)
        assert message.is_ready_for_retry()

        # Set future retry time - should not be ready
        future_time = datetime.utcnow() + timedelta(seconds=10)
        message.next_retry = future_time
        assert not message.is_ready_for_retry()

        # Set past retry time - should be ready
        past_time = datetime.utcnow() - timedelta(seconds=1)
        message.next_retry = past_time
        assert message.is_ready_for_retry()


class TestQueueConfig:
    """Test cases for QueueConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = QueueConfig(
            max_size=2000,
            persistence_enabled=True,
            cleanup_interval=3600,
            max_age_hours=48,
        )

        assert config.max_size == 2000
        assert config.persistence_enabled is True
        assert config.cleanup_interval == 3600
        assert config.max_age_hours == 48

    def test_invalid_max_size(self):
        """Test invalid max_size validation."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            QueueConfig(max_size=0)

    def test_invalid_cleanup_interval(self):
        """Test invalid cleanup_interval validation."""
        with pytest.raises(ValueError, match="cleanup_interval must be positive"):
            QueueConfig(cleanup_interval=-1)

    def test_invalid_max_age_hours(self):
        """Test invalid max_age_hours validation."""
        with pytest.raises(ValueError, match="max_age_hours must be positive"):
            QueueConfig(max_age_hours=0)


class TestInMemoryMessageStorage:
    """Test cases for in-memory message storage."""

    def test_storage_initialization(self):
        """Test storage initialization."""
        storage = InMemoryMessageStorage()
        assert len(storage.get_pending_messages()) == 0

    def test_store_message(self):
        """Test storing a message."""
        storage = InMemoryMessageStorage()
        message = QueuedMessage(content="Test", webhook_url="https://discord.com/test")

        message_id = storage.store_message(message)
        assert isinstance(message_id, str)
        assert len(message_id) > 0

        # Verify message is stored
        pending = storage.get_pending_messages()
        assert len(pending) == 1
        assert pending[0].content == "Test"

    def test_get_pending_messages_priority_order(self):
        """Test messages returned in priority order."""
        storage = InMemoryMessageStorage()

        # Store messages with different priorities
        low_msg = QueuedMessage("Low", "https://discord.com/test", MessagePriority.LOW)
        high_msg = QueuedMessage(
            "High", "https://discord.com/test", MessagePriority.HIGH
        )
        normal_msg = QueuedMessage(
            "Normal", "https://discord.com/test", MessagePriority.NORMAL
        )

        storage.store_message(low_msg)
        storage.store_message(high_msg)
        storage.store_message(normal_msg)

        pending = storage.get_pending_messages()
        assert len(pending) == 3

        # Should be ordered: HIGH, NORMAL, LOW
        assert pending[0].content == "High"
        assert pending[1].content == "Normal"
        assert pending[2].content == "Low"

    def test_update_message(self):
        """Test updating a stored message."""
        storage = InMemoryMessageStorage()
        message = QueuedMessage("Test", "https://discord.com/test")
        message_id = storage.store_message(message)

        # Update message
        message.retry_count = 1
        storage.update_message(message_id, message)

        # Verify update
        pending = storage.get_pending_messages()
        assert pending[0].retry_count == 1

    def test_remove_message(self):
        """Test removing a message."""
        storage = InMemoryMessageStorage()
        message = QueuedMessage("Test", "https://discord.com/test")
        message_id = storage.store_message(message)

        # Remove message
        removed = storage.remove_message(message_id)
        assert removed is True

        # Verify removal
        assert len(storage.get_pending_messages()) == 0

        # Try to remove non-existent message
        removed = storage.remove_message("nonexistent")
        assert removed is False

    def test_cleanup_old_messages(self):
        """Test cleanup of old messages."""
        storage = InMemoryMessageStorage()

        # Create old message (simulate by setting created_at in past)
        old_message = QueuedMessage("Old", "https://discord.com/test")
        old_message.created_at = datetime.utcnow() - timedelta(hours=25)

        # Create recent message
        recent_message = QueuedMessage("Recent", "https://discord.com/test")

        storage.store_message(old_message)
        storage.store_message(recent_message)

        # Cleanup messages older than 24 hours
        cleanup_count = storage.cleanup_old_messages(max_age_hours=24)

        assert cleanup_count == 1
        pending = storage.get_pending_messages()
        assert len(pending) == 1
        assert pending[0].content == "Recent"


class TestSqliteMessageStorage:
    """Test cases for SQLite message storage."""

    def test_storage_initialization(self):
        """Test SQLite storage initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_queue.db")
            storage = SqliteMessageStorage(db_path)

            # Database file should be created
            assert os.path.exists(db_path)

            # Should have no messages initially
            assert len(storage.get_pending_messages()) == 0

    def test_store_and_retrieve_message(self):
        """Test storing and retrieving messages from SQLite."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_queue.db")
            storage = SqliteMessageStorage(db_path)

            message = QueuedMessage(
                content="Test SQLite",
                webhook_url="https://discord.com/test",
                priority=MessagePriority.HIGH,
            )

            message_id = storage.store_message(message)
            assert isinstance(message_id, str)

            # Retrieve and verify
            pending = storage.get_pending_messages()
            assert len(pending) == 1
            retrieved = pending[0]

            assert retrieved.content == "Test SQLite"
            assert retrieved.webhook_url == "https://discord.com/test"
            assert retrieved.priority == MessagePriority.HIGH

    def test_sqlite_persistence(self):
        """Test that SQLite storage persists across instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_persistence.db")

            # Store message in first instance
            storage1 = SqliteMessageStorage(db_path)
            message = QueuedMessage("Persistent", "https://discord.com/test")
            message_id = storage1.store_message(message)

            # Create new instance and verify persistence
            storage2 = SqliteMessageStorage(db_path)
            pending = storage2.get_pending_messages()

            assert len(pending) == 1
            assert pending[0].content == "Persistent"

    def test_sqlite_update_message(self):
        """Test updating message in SQLite."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_update.db")
            storage = SqliteMessageStorage(db_path)

            message = QueuedMessage("Test", "https://discord.com/test")
            message_id = storage.store_message(message)

            # Update retry count
            message.retry_count = 2
            message.next_retry = datetime.utcnow() + timedelta(seconds=30)

            storage.update_message(message_id, message)

            # Verify update
            pending = storage.get_pending_messages()
            assert pending[0].retry_count == 2
            assert pending[0].next_retry is not None

    def test_sqlite_cleanup(self):
        """Test SQLite cleanup functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_cleanup.db")
            storage = SqliteMessageStorage(db_path)

            # Add old and new messages
            old_message = QueuedMessage("Old", "https://discord.com/test")
            old_message.created_at = datetime.utcnow() - timedelta(hours=25)

            new_message = QueuedMessage("New", "https://discord.com/test")

            storage.store_message(old_message)
            storage.store_message(new_message)

            # Cleanup old messages
            cleanup_count = storage.cleanup_old_messages(max_age_hours=24)

            assert cleanup_count == 1
            pending = storage.get_pending_messages()
            assert len(pending) == 1
            assert pending[0].content == "New"


class TestMessageQueue:
    """Test cases for MessageQueue functionality."""

    def test_queue_initialization(self):
        """Test message queue initialization."""
        config = QueueConfig()
        storage = InMemoryMessageStorage()
        queue = MessageQueue(config, storage)

        assert queue.size() == 0
        assert not queue.is_full()

    @pytest.mark.asyncio
    async def test_enqueue_message(self):
        """Test enqueueing messages."""
        config = QueueConfig(max_size=2)
        storage = InMemoryMessageStorage()
        queue = MessageQueue(config, storage)

        message = QueuedMessage("Test", "https://discord.com/test")

        # Enqueue message
        message_id = await queue.enqueue(message)
        assert isinstance(message_id, str)
        assert queue.size() == 1

    @pytest.mark.asyncio
    async def test_enqueue_full_queue(self):
        """Test enqueueing to full queue."""
        config = QueueConfig(max_size=1)
        storage = InMemoryMessageStorage()
        queue = MessageQueue(config, storage)

        # Fill queue
        message1 = QueuedMessage("Test1", "https://discord.com/test")
        await queue.enqueue(message1)
        assert queue.is_full()

        # Try to enqueue to full queue
        message2 = QueuedMessage("Test2", "https://discord.com/test")
        with pytest.raises(RuntimeError, match="Queue is full"):
            await queue.enqueue(message2)

    @pytest.mark.asyncio
    async def test_dequeue_message(self):
        """Test dequeueing messages in priority order."""
        config = QueueConfig()
        storage = InMemoryMessageStorage()
        queue = MessageQueue(config, storage)

        # Enqueue messages with different priorities
        low_msg = QueuedMessage("Low", "https://discord.com/test", MessagePriority.LOW)
        high_msg = QueuedMessage(
            "High", "https://discord.com/test", MessagePriority.HIGH
        )

        await queue.enqueue(low_msg)
        await queue.enqueue(high_msg)

        # Dequeue should return high priority first
        dequeued = await queue.dequeue()
        assert dequeued.content == "High"
        assert queue.size() == 1

    @pytest.mark.asyncio
    async def test_dequeue_empty_queue(self):
        """Test dequeueing from empty queue."""
        config = QueueConfig()
        storage = InMemoryMessageStorage()
        queue = MessageQueue(config, storage)

        # Dequeue from empty queue should return None
        dequeued = await queue.dequeue()
        assert dequeued is None

    @pytest.mark.asyncio
    async def test_dequeue_retry_timing(self):
        """Test dequeue respects retry timing."""
        config = QueueConfig()
        storage = InMemoryMessageStorage()
        queue = MessageQueue(config, storage)

        # Create message with future retry time
        message = QueuedMessage("Test", "https://discord.com/test")
        message.next_retry = datetime.utcnow() + timedelta(seconds=10)
        message.retry_count = 1

        await queue.enqueue(message)

        # Should not dequeue message that's not ready for retry
        dequeued = await queue.dequeue()
        assert dequeued is None

    @pytest.mark.asyncio
    async def test_requeue_message(self):
        """Test requeueing failed messages."""
        config = QueueConfig()
        storage = InMemoryMessageStorage()
        queue = MessageQueue(config, storage)

        message = QueuedMessage("Test", "https://discord.com/test", max_retries=3)
        message_id = await queue.enqueue(message)

        # Dequeue message
        dequeued = await queue.dequeue()

        # Requeue with delay
        await queue.requeue(dequeued, delay_seconds=5)

        # Message should be back in queue with incremented retry count
        assert queue.size() == 1

        # Get the message (directly from storage for testing)
        pending = storage.get_pending_messages()
        requeued_msg = pending[0]
        assert requeued_msg.retry_count == 1
        assert requeued_msg.next_retry is not None

    @pytest.mark.asyncio
    async def test_requeue_max_retries_exceeded(self):
        """Test requeueing when max retries exceeded."""
        config = QueueConfig()
        storage = InMemoryMessageStorage()
        queue = MessageQueue(config, storage)

        message = QueuedMessage("Test", "https://discord.com/test", max_retries=1)
        message.retry_count = 1  # Already at max
        await queue.enqueue(message)

        dequeued = await queue.dequeue()

        # Should not requeue - should remove instead
        await queue.requeue(dequeued, delay_seconds=5)

        # Message should be removed from queue
        assert queue.size() == 0

    @pytest.mark.asyncio
    async def test_remove_message(self):
        """Test removing specific message from queue."""
        config = QueueConfig()
        storage = InMemoryMessageStorage()
        queue = MessageQueue(config, storage)

        message = QueuedMessage("Test", "https://discord.com/test")
        message_id = await queue.enqueue(message)

        # Remove message
        removed = await queue.remove(message_id)
        assert removed is True
        assert queue.size() == 0

    @pytest.mark.asyncio
    async def test_cleanup_old_messages(self):
        """Test automatic cleanup of old messages."""
        config = QueueConfig(max_age_hours=24)
        storage = InMemoryMessageStorage()
        queue = MessageQueue(config, storage)

        # Create old message
        old_message = QueuedMessage("Old", "https://discord.com/test")
        old_message.created_at = datetime.utcnow() - timedelta(hours=25)

        # Create new message
        new_message = QueuedMessage("New", "https://discord.com/test")

        await queue.enqueue(old_message)
        await queue.enqueue(new_message)

        # Run cleanup
        cleanup_count = await queue.cleanup_old_messages()

        assert cleanup_count == 1
        assert queue.size() == 1


class TestMessageQueueManager:
    """Test cases for MessageQueueManager."""

    def test_manager_initialization(self):
        """Test queue manager initialization."""
        config = QueueConfig()
        manager = MessageQueueManager(config)

        assert manager.get_queue_size() == 0
        assert not manager.is_queue_full()

    @pytest.mark.asyncio
    async def test_manager_lifecycle(self):
        """Test manager startup and shutdown."""
        config = QueueConfig()
        manager = MessageQueueManager(config)

        # Start manager
        await manager.start()

        # Should be able to enqueue messages
        message = QueuedMessage("Test", "https://discord.com/test")
        message_id = await manager.enqueue_message(message)
        assert isinstance(message_id, str)

        # Stop manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_manager_auto_cleanup(self):
        """Test manager automatic cleanup functionality."""
        # Use short cleanup interval for testing
        config = QueueConfig(cleanup_interval=1, max_age_hours=1)
        manager = MessageQueueManager(config)

        await manager.start()

        try:
            # Add old message
            old_message = QueuedMessage("Old", "https://discord.com/test")
            old_message.created_at = datetime.utcnow() - timedelta(hours=2)
            await manager.enqueue_message(old_message)

            # Wait for cleanup cycle
            await asyncio.sleep(1.5)

            # Old message should be cleaned up
            assert manager.get_queue_size() == 0

        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_manager_callback_registration(self):
        """Test callback registration for queue events."""
        config = QueueConfig()
        manager = MessageQueueManager(config)

        # Register callbacks
        enqueue_callback = Mock()
        dequeue_callback = Mock()

        manager.register_callback("message_enqueued", enqueue_callback)
        manager.register_callback("message_dequeued", dequeue_callback)

        await manager.start()

        try:
            # Enqueue message - should trigger callback
            message = QueuedMessage("Test", "https://discord.com/test")
            await manager.enqueue_message(message)

            # Process message - should trigger dequeue callback
            processed = await manager.dequeue_message()

            # Verify callbacks were called
            enqueue_callback.assert_called_once()
            dequeue_callback.assert_called_once()

        finally:
            await manager.stop()


class TestFactoryFunction:
    """Test cases for create_message_queue_manager factory."""

    def test_create_with_defaults(self):
        """Test creating manager with default configuration."""
        manager = create_message_queue_manager()
        assert isinstance(manager, MessageQueueManager)
        assert manager.get_queue_size() == 0

    def test_create_with_custom_config(self):
        """Test creating manager with custom configuration."""
        config = QueueConfig(max_size=500, persistence_enabled=False)
        manager = create_message_queue_manager(config)
        assert isinstance(manager, MessageQueueManager)

    def test_create_with_persistence(self):
        """Test creating manager with SQLite persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_factory.db")

            config = QueueConfig(persistence_enabled=True)
            manager = create_message_queue_manager(config, db_path)

            assert isinstance(manager, MessageQueueManager)
            # Should create database file
            assert os.path.exists(db_path)
