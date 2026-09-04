from rest_framework import serializers

from .models import AuditLogEntry, Detection, DetectionJob


class DetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Detection
        fields = "__all__"


class AuditLogEntrySerializer(serializers.ModelSerializer):
    class Meta:
        model = AuditLogEntry
        fields = "__all__"


class DetectionJobSerializer(serializers.ModelSerializer):
    detections = DetectionSerializer(many=True, read_only=True)

    class Meta:
        model = DetectionJob
        fields = [
            "id",
            "input_path",
            "status",
            "progress",
            "error_message",
            "created_at",
            "started_at",
            "completed_at",
            "detections",
        ]