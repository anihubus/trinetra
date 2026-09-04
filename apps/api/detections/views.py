import uuid

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import DetectionJob
from .serializers import DetectionJobSerializer
from .tasks import run_detection_job


class JobListCreateView(APIView):
    def get(self, request):
        jobs = DetectionJob.objects.order_by("-created_at")
        serializer = DetectionJobSerializer(jobs, many=True)
        return Response(serializer.data)

    def post(self, request):
        input_path = request.data.get("input_path")

        if not input_path:
            return Response(
                {"error": "input_path is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        job = DetectionJob.objects.create(
            id=uuid.uuid4(),
            input_path=input_path,
        )

        run_detection_job.delay(str(job.id))

        serializer = DetectionJobSerializer(job)

        return Response(
            serializer.data,
            status=status.HTTP_202_ACCEPTED,
        )


class JobDetailView(APIView):
    def get(self, request, job_id):
        try:
            job = DetectionJob.objects.get(id=job_id)
        except DetectionJob.DoesNotExist:
            return Response(
                {"error": "Job not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = DetectionJobSerializer(job)
        return Response(serializer.data)