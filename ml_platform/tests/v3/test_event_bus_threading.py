"""The event bus must wake a *parked* consumer when published from a thread.

sklearn training runs in a ThreadPoolExecutor worker and calls
``event_bus.publish`` directly from there, while ``/ws/logs/{id}`` is a
coroutine parked on ``await queue.get()``. ``asyncio.Queue`` is not
thread-safe: a bare ``put_nowait`` from off-loop wakes the consumer via
``loop.call_soon``, which does not interrupt a sleeping event loop. Nothing is
lost — the entry just sits there until unrelated traffic wakes the loop, which
is why live training logs looked frozen and then arrived in a burst.

Getting this test to actually fail on the old code takes care:

* the consumer must already be **parked** before the publish. ``Queue.get()``
  returns immediately when an item is already queued, so publishing first
  exercises a fast path that works either way.
* the loop must then be **idle**. Any pending short timer or request wakes it
  for its own reasons and delivers the backlog as a side effect, hiding the bug.

So these assert delivery *latency*, not eventual delivery.
"""
import asyncio
import threading
import time

import pytest

from app.core.logger import EventBus

CHANNEL = "logs:some-task"
# The fixed path delivers in well under a millisecond; the broken one stalls
# until the loop wakes for another reason (~2.9s when measured). IDLE_SECONDS
# is how long the loop is left with nothing to do, and must exceed the
# threshold so a stalled delivery is actually late rather than merely last.
MAX_DELIVERY_SECONDS = 0.5
PUBLISH_DELAY_SECONDS = 0.3   # long enough for the loop to be inside select()
IDLE_SECONDS = 2.0            # must exceed PUBLISH_DELAY + MAX_DELIVERY


async def test_publish_from_worker_thread_wakes_a_parked_consumer():
    bus = EventBus()
    queue = bus.subscribe(CHANNEL)
    received_at: list[float] = []

    async def consumer() -> None:
        await queue.get()
        received_at.append(time.perf_counter())

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.2)          # let the consumer actually park on get()
    assert not received_at

    # The publish must land while the loop is *already* asleep inside select().
    # Publishing before the await instead leaves the wakeup sitting in the
    # loop's _ready list, which the very next _run_once drains with timeout=0 —
    # delivering promptly even on the broken implementation and turning this
    # into a test that cannot fail. So the thread waits for the loop to park,
    # mirroring a trainer that emits a log line partway through a fit.
    published_at: list[float] = []

    def publish_after_the_loop_parks() -> None:
        time.sleep(PUBLISH_DELAY_SECONDS)
        published_at.append(time.perf_counter())
        bus.publish(CHANNEL, {"type": "log", "message": "from a worker thread"})

    thread = threading.Thread(target=publish_after_the_loop_parks)
    thread.start()

    # Nothing else is scheduled: only a publish-side wakeup can deliver early.
    await asyncio.sleep(IDLE_SECONDS)
    thread.join()
    task.cancel()

    assert received_at, "log never reached the consumer at all"
    latency = received_at[0] - published_at[0]
    assert latency < MAX_DELIVERY_SECONDS, (
        f"log took {latency:.3f}s to reach a parked consumer on an idle loop; "
        "the event loop is not being woken on publish"
    )


async def test_publish_on_the_loop_thread_still_works():
    """The same-loop path must not regress while fixing the cross-thread one."""
    bus = EventBus()
    queue = bus.subscribe(CHANNEL)
    bus.publish(CHANNEL, {"type": "log", "message": "same loop"})
    message = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert message["message"] == "same loop"


async def test_unsubscribe_stops_delivery():
    """Guards the (queue, loop) bookkeeping introduced for thread-safety."""
    bus = EventBus()
    queue = bus.subscribe(CHANNEL)
    bus.unsubscribe(CHANNEL, queue)
    bus.publish(CHANNEL, {"type": "log", "message": "should not arrive"})
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.get(), timeout=0.2)
