/**
 * api.js  — Frontend ↔ FastAPI service layer
 *
 * All calls go through these functions. Swap BASE_URL for production.
 */

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

async function post(path, body) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }

  return res.json();
}

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }

  return res.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// Translation  →  /api/translate
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Full pipeline: text → gloss → IDs → model → keypoint frames
 *
 * @param {string} text  Raw English sentence
 * @returns {Promise<TranslationResponse>}
 *
 * TranslationResponse shape:
 * {
 *   original_text: string,
 *   glosses:       string[],
 *   gloss_ids:     number[],
 *   oov_glosses:   string[],
 *   frames:        KeypointFrame[],   // [{joints: {nose: {x,y}, ...}}, ...]
 *   frame_count:   number,
 *   fps:           number,
 *   duration_ms:   number,
 *   processing_ms: number,
 * }
 */
export async function translateText(text) {
  return post("/translate", { text });
}

/**
 * Preview gloss + IDs without running the model (useful for debug UI)
 */
export async function previewGloss(text) {
  return post("/translate/preview-gloss", { text });
}

// ─────────────────────────────────────────────────────────────────────────────
// Stories  →  /api/stories
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch all story summaries for the Stories listing page.
 */
export async function getStories() {
  return get("/stories");
}

/**
 * Fetch a single story with pre-generated keypoint frames.
 *
 * @param {string} storyId
 * @returns {Promise<StoryDetail>}
 */
export async function getStoryKeypoints(storyId) {
  return get(`/stories/${storyId}`);
}