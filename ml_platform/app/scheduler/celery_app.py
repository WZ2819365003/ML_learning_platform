"""
Celery application configuration for the ML Platform V3 scheduler.

Start worker:
    cd ml_platform
    celery -A app.scheduler.celery_app worker --loglevel=info --concurrency=4

Or with multiple queues:
    celery -A app.scheduler.celery_app worker -Q train,explain,default --loglevel=info
"""

from __future__ import annotations

from celery import Celery

from app.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "ml_platform",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
    include=["app.scheduler.celery_tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Time limits
    task_soft_time_limit=3600,   # 60 min soft limit — sends SoftTimeLimitExceeded
    task_time_limit=3900,        # 65 min hard kill
    # Retry behaviour
    task_acks_late=True,         # ack only after success — enables re-queue on worker crash
    task_reject_on_worker_lost=True,
    # Result expiry (keep results 7 days in Redis)
    result_expires=604800,
    # Queue routing
    task_default_queue="default",
    task_queues={
        "train":   {"exchange": "train",   "routing_key": "train"},
        "explain": {"exchange": "explain", "routing_key": "explain"},
        "default": {"exchange": "default", "routing_key": "default"},
    },
    task_routes={
        "app.scheduler.celery_tasks.run_train_task":   {"queue": "train"},
        "app.scheduler.celery_tasks.run_explain_task": {"queue": "explain"},
    },
    # Timezone
    timezone="UTC",
    enable_utc=True,
)
