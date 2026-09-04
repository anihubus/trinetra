from django.http import JsonResponse
from django.urls import include, path


def health(request):
    return JsonResponse(
        {
            "status": "ok",
            "service": "drishti-api",
        }
    )


urlpatterns = [
    path("api/health/", health),
    path("api/", include("detections.urls")),
]