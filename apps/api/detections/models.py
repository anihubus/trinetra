import uuid

from django.db import models


class DetectionJob(models.Model):
    STATUS = [("queued", "queued"), ("running", "running"),
              ("completed", "completed"), ("failed", "failed")]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source_file = models.CharField(max_length=255)          # original log/image name
    input_path = models.CharField(max_length=1024)          # where it landed on disk
    xtf_path = models.CharField(max_length=1024, blank=True, default="")
    nav_path = models.CharField(max_length=1024, blank=True, default="")
    status = models.CharField(max_length=16, choices=STATUS, default="queued")
    progress = models.FloatField(default=0.0)
    celery_task_id = models.CharField(max_length=128, blank=True, default="")
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)


class Detection(models.Model):
    detection_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(DetectionJob, related_name="detections", on_delete=models.CASCADE)

    ping_id = models.CharField(max_length=255, blank=True, default="")
    timestamp = models.DateTimeField(null=True, blank=True)

    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    class_label = models.CharField(max_length=64)
    confidence_score = models.FloatField()                  # 0–100, calibrated
    bounding_geometry = models.JSONField(default=dict)      # {bbox, mask_polygon, width_m, height_m}

    across_track_m = models.FloatField(null=True, blank=True)
    side = models.CharField(max_length=12, blank=True, default="")

    review_status = models.CharField(max_length=24, default="pending_review")
    source_file = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["job", "review_status"])]


class AuditLogEntry(models.Model):
    job = models.ForeignKey(DetectionJob, related_name="audit", on_delete=models.CASCADE)
    detection = models.ForeignKey(Detection, related_name="audit", null=True, blank=True,
                                 on_delete=models.CASCADE)
    action = models.CharField(max_length=255)
    actor = models.CharField(max_length=128, blank=True, default="")
    details = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
