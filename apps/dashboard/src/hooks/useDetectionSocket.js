// React hook: subscribes to a job's WebSocket channel, incrementally accumulates
// detections into state as each tile event arrives -- this is what makes
// MapView and ImageOverlay populate live instead of waiting for the full job.
