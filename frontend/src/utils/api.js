const API_HOST = import.meta.env.VITE_API_HOST?.replace(/\/$/, "") || "/api";
const ANALYZE_VIDEO_URL = `${API_HOST}/video/analyze`;
const VIDEO_JOBS_URL = `${API_HOST}/video/jobs`;

const readErrorMessage = async (response) => {
  const fallback = `Backend returned ${response.status}`;
  const errorText = await response.text();

  if (!errorText) return fallback;

  try {
    const errorPayload = JSON.parse(errorText);
    return errorPayload.detail || fallback;
  } catch {
    return errorText;
  }
};

export const analyzeVideo = async (file, signal) => {
  const formData = new FormData();
  formData.append('video', file);

  const response = await fetch(ANALYZE_VIDEO_URL, {
    method: 'POST',
    body: formData,
    signal,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.text();
};

export const createVideoJob = async (file, signal) => {
  const formData = new FormData();
  formData.append('video', file);

  const response = await fetch(VIDEO_JOBS_URL, {
    method: 'POST',
    body: formData,
    signal,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
};

export const getVideoJob = async (jobId, signal) => {
  const response = await fetch(`${VIDEO_JOBS_URL}/${jobId}`, { signal });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
};

export const getVideoJobCsv = async (jobId, signal) => {
  const response = await fetch(`${VIDEO_JOBS_URL}/${jobId}/csv`, { signal });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.text();
};
