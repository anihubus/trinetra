# DRISHTI -- Architecture Notes

See the two coach-session diagrams for the full picture:
  1. Real-time processing architecture (React -> Django -> Celery worker -> WebSocket push -> PostgreSQL)
  2. Training -> export -> edge deployment pipeline (dataset -> YOLOv8n-seg -> ONNX/INT8 -> CPU or Jetson)

Key principle carried through the whole codebase: the client (frontend) never talks
to the ML model directly. Everything routes through the Django REST API / WebSocket
layer, which is what lets ml/ be retrained or edge/ be redeployed without touching
frontend/ at all -- see docs/api_contract.md for the exact interface that guarantees this.
