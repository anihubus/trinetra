from pathlib import Path
import sys

from decouple import Csv, config


BASE_DIR = Path(__file__).resolve().parent.parent

# Sonar-Drishti/
REPO_ROOT = BASE_DIR.parent.parent

# Makes this possible:
# from ml.inference.pipeline import run_pipeline
sys.path.insert(0, str(REPO_ROOT))

# Required by some ML imports.
sys.path.insert(0, str(REPO_ROOT / "ml" / "scripts"))


SECRET_KEY = config(
    "SECRET_KEY",
    default="dev-insecure-change-me",
)

DEBUG = config(
    "DEBUG",
    default=True,
    cast=bool,
)

ALLOWED_HOSTS = config(
    "ALLOWED_HOSTS",
    default="localhost,127.0.0.1",
    cast=Csv(),
)


INSTALLED_APPS = [
    "daphne",

    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",

    "rest_framework",
    "corsheaders",
    "drf_spectacular",
    "channels",

    "detections",
]


MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]


ROOT_URLCONF = "drishti_api.urls"

WSGI_APPLICATION = "drishti_api.wsgi.application"

ASGI_APPLICATION = "drishti_api.asgi.application"


TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {},
    }
]


# --------------------------------------------------
# PostgreSQL
# --------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",

        "NAME": config(
            "POSTGRES_DB",
            default="drishti",
        ),

        "USER": config(
            "POSTGRES_USER",
            default="drishti",
        ),

        "PASSWORD": config(
            "POSTGRES_PASSWORD",
            default="drishti",
        ),

        "HOST": config(
            "POSTGRES_HOST",
            default="localhost",
        ),

        "PORT": config(
            "POSTGRES_PORT",
            default="5432",
        ),
    }
}


# --------------------------------------------------
# Redis
# --------------------------------------------------

REDIS_URL = config(
    "REDIS_URL",
    default="redis://localhost:6379/0",
)


# --------------------------------------------------
# Django Channels / WebSockets
# --------------------------------------------------

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",

        "CONFIG": {
            "hosts": [REDIS_URL],
        },
    }
}


# --------------------------------------------------
# Celery
# --------------------------------------------------

CELERY_BROKER_URL = REDIS_URL

CELERY_RESULT_BACKEND = REDIS_URL

CELERY_TASK_TRACK_STARTED = True


# --------------------------------------------------
# DRF
# --------------------------------------------------

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS":
        "drf_spectacular.openapi.AutoSchema",

    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
}


SPECTACULAR_SETTINGS = {
    "TITLE": "DRISHTI API",
    "VERSION": "1.0",
}


# --------------------------------------------------
# CORS
# --------------------------------------------------

CORS_ALLOWED_ORIGINS = config(
    "CORS_ALLOWED_ORIGINS",
    default="http://localhost:5173",
    cast=Csv(),
)


# --------------------------------------------------
# Static / uploaded files
# --------------------------------------------------

STATIC_URL = "static/"

STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"

MEDIA_ROOT = config(
    "MEDIA_ROOT",
    default=str(BASE_DIR / "media"),
)


DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# --------------------------------------------------
# ML model paths
# --------------------------------------------------

DRISHTI_MODEL = config(
    "DRISHTI_MODEL",
    default=str(
        REPO_ROOT
        / "ml"
        / "models"
        / "exported"
        / "best_detector.onnx"
    ),
)


DRISHTI_CALIBRATOR = config(
    "DRISHTI_CALIBRATOR",
    default=str(
        REPO_ROOT
        / "ml"
        / "models"
        / "exported"
        / "calibrator.pkl"
    ),
)