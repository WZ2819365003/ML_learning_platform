from __future__ import annotations

import asyncio
import inspect
import queue


class FakePubSub:
    def __init__(self, messages: queue.Queue):
        self.messages = messages
        self.closed = False
        self.pattern = None

    def psubscribe(self, pattern):
        self.pattern = pattern

    def get_message(self, **_kwargs):
        try:
            return self.messages.get(timeout=0.05)
        except queue.Empty:
            return None

    def close(self):
        self.closed = True


class FakeRedis:
    def __init__(self):
        self.messages: queue.Queue = queue.Queue()
        self.published: list[tuple[str, str]] = []
        self.pubsubs: list[FakePubSub] = []

    def publish(self, channel, payload):
        self.published.append((channel, payload))
        self.messages.put({"type": "pmessage", "channel": channel, "data": payload})
        return 1

    def pubsub(self, **_kwargs):
        pubsub = FakePubSub(self.messages)
        self.pubsubs.append(pubsub)
        return pubsub


class FailingSubscribePubSub(FakePubSub):
    def psubscribe(self, pattern):
        raise ConnectionError("temporary disconnect")


class ReconnectingFakeRedis(FakeRedis):
    def pubsub(self, **_kwargs):
        if not self.pubsubs:
            pubsub = FailingSubscribePubSub(self.messages)
        else:
            pubsub = FakePubSub(self.messages)
        self.pubsubs.append(pubsub)
        return pubsub


async def test_redis_event_bus_publish_is_sync_and_bridge_fans_out():
    from app.core.event_bus_redis import RedisEventBus

    fake = FakeRedis()
    bus = RedisEventBus(redis_client=fake, queue_maxsize=2, reconnect_delay=0.01)
    assert not inspect.iscoroutinefunction(bus.publish)

    local_queue = bus.subscribe("training:task-1")
    await bus.start()
    await asyncio.to_thread(bus.publish, "training:task-1", {"step": 1})
    message = await asyncio.wait_for(local_queue.get(), timeout=1)

    assert message == {"step": 1}
    assert fake.published[0][0] == "ml-platform:training:task-1"
    bus.unsubscribe("training:task-1", local_queue)
    await bus.stop()
    assert all(pubsub.closed for pubsub in fake.pubsubs)


async def test_redis_event_bus_drops_oldest_when_local_queue_is_full():
    from app.core.event_bus_redis import RedisEventBus

    fake = FakeRedis()
    bus = RedisEventBus(redis_client=fake, queue_maxsize=2, reconnect_delay=0.01)
    local_queue = bus.subscribe("logs:task-2")
    await bus.start()
    for sequence in range(3):
        bus.publish("logs:task-2", {"sequence": sequence})

    async def wait_until_full():
        for _ in range(100):
            if local_queue.qsize() == 2:
                return
            await asyncio.sleep(0.01)
        raise AssertionError("bridge did not fill the local queue")

    await wait_until_full()
    assert [local_queue.get_nowait(), local_queue.get_nowait()] == [
        {"sequence": 1},
        {"sequence": 2},
    ]
    await bus.stop()


async def test_redis_event_bus_reconnects_after_subscription_failure():
    from app.core.event_bus_redis import RedisEventBus

    fake = ReconnectingFakeRedis()
    bus = RedisEventBus(redis_client=fake, reconnect_delay=0.01)
    local_queue = bus.subscribe("training:reconnect")
    await bus.start()

    for _ in range(100):
        if len(fake.pubsubs) >= 2:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("bridge did not reconnect")

    bus.publish("training:reconnect", {"connected": True})
    assert await asyncio.wait_for(local_queue.get(), timeout=1) == {"connected": True}
    await bus.stop()
