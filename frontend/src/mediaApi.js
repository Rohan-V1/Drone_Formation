export const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

export async function submitMedia(file, kind) {
  const endpoint = kind === "image" ? "/api/analyze-image" : "/api/analyze";
  const response = await fetch(`${API}${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": file.type || "application/octet-stream",
      "X-File-Name": encodeURIComponent(file.name),
    },
    body: file,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || payload.error || `Analysis failed (${response.status})`);
  return payload;
}
