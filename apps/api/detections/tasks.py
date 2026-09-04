import uuid

from celery import shared_task
from django.utils import timezone

from .models import AuditLogEntry, Detection, DetectionJob


@shared_task
def architecture_test():
    return "DRISHTI Celery works"


@shared_task(bind=True)
def run_detection_job(self, job_id):
    job = DetectionJob.objects.get(id=job_id)

    job.status = "running"
    job.started_at = timezone.now()
    job.progress = 0.0
    job.save(
        update_fields=[
            "status",
            "started_at",
            "progress",
        ]
    )

    try:
        # ---------------------------------------------------------
        # ML INTEGRATION SEAM
        # ---------------------------------------------------------
        #
        # Later the real worker will call:
        #
        # from ml.inference.pipeline import run_pipeline
        #
        # report = run_pipeline(...)
        #
        # We are NOT executing the actual ML pipeline yet.
        # ---------------------------------------------------------

        job.progress = 0.5
        job.save(update_fields=["progress"])

        job.status = "completed"
        job.progress = 1.0
        job.completed_at = timezone.now()
        job.save(
            update_fields=[
                "status",
                "progress",
                "completed_at",
            ]
        )

        AuditLogEntry.objects.create(
            job=job,
            action="detection.completed",
            details={
                "message": "Detection job completed successfully."
            },
        )

        return {
            "job_id": str(job.id),
            "status": "completed",
        }

    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)
        job.completed_at = timezone.now()
        job.save(
            update_fields=[
                "status",
                "error_message",
                "completed_at",
            ]
        )

        AuditLogEntry.objects.create(
            job=job,
            action="detection.failed",
            details={"error": str(exc)},
        )

        raise