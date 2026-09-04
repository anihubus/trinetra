export function connectToJob(jobId, handlers = {}) {
  const base =
    import.meta.env.VITE_WS_BASE ||
    `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`;

  const socket = new WebSocket(
    `${base}/ws/jobs/${jobId}/`
  );

  socket.onopen = () => {
    handlers.onOpen?.();
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      console.log("WebSocket message:", data);

      if (data.type === "detection.partial") {
        handlers.onPartial?.(data);
      }

      if (data.type === "detection.complete") {
        handlers.onComplete?.(data);
      }

      if (data.type === "detection.failed") {
        handlers.onFailed?.(data);
      }
    } catch (error) {
      console.error(
        "Failed to parse WebSocket message:",
        error
      );
    }
  };

  socket.onerror = (error) => {
    console.error("WebSocket error:", error);
    handlers.onError?.(error);
  };

  socket.onclose = (event) => {
    console.log("WebSocket closed:", event.code);
    handlers.onClose?.();
  };

  return socket;
}