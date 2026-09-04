import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || "/api",
});

export async function getJobs() {
  const response = await api.get("/jobs/");
  return response.data;
}

export async function createJob(inputPath) {
  const response = await api.post("/jobs/", {
    input_path: inputPath,
  });

  return response.data;
}

export async function getJob(jobId) {
  const response = await api.get(`/jobs/${jobId}/`);
  return response.data;
}