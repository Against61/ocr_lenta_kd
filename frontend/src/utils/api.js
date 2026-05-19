const ANALYZE_VIDEO_URL = 'http://localhost:8041/video/analyze';

export const analyzeVideo = async (file, signal) => {
  const formData = new FormData();
  formData.append('video', file);

  const response = await fetch(ANALYZE_VIDEO_URL, {
    method: 'POST',
    body: formData,
    signal,
  });

  if (!response.ok) {
    let message = `Backend returned ${response.status}`;

    try {
      const errorPayload = await response.json();
      message = errorPayload.detail || message;
    } catch {
      const errorText = await response.text();
      message = errorText || message;
    }

    throw new Error(message);
  }

  return response.text();
};
