import { useState } from "react";
import SkeletonViewer from "../components/SkeletonViewer";
import { translateText } from "../services/api";
import "./Translate.css";

export default function Translate() {
  const [text, setText]         = useState("");
  const [frames, setFrames]     = useState([]);   // ← was: keypoints
  const [fps, setFps]           = useState(25);
  const [glosses, setGlosses]   = useState([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  const handleTranslate = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setFrames([]);
    setGlosses([]);
    try {
      const data = await translateText(text);
      // data.frames   → [{joints: {nose:{x,y,z}, ...}}, ...]
      // data.fps      → 25
      // data.glosses  → ["HELLO", "HOW", "YOU"]
      setFrames(data.frames);
      setFps(data.fps);
      setGlosses(data.glosses);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="translate-page">
      <span className="section-tag">✏️ Translate</span>
      <h2 className="section-title">Text → ASL Animation</h2>
      <p className="section-sub">
        Enter any sentence and let Signtale animate it in American Sign Language.
      </p>

      <div className="translate-grid">
        {/* Input panel */}
        <div className="translate-left">
          <div className="translate-box">
            <label className="translate-label">Your text</label>
            <textarea
              className="translate-textarea"
              placeholder="e.g. I am happy to meet you"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleTranslate()}
            />
            <div className="translate-actions">
              <span className="char-count">{text.length} chars</span>
              <button
                className="btn-translate"
                onClick={handleTranslate}
                disabled={!text.trim() || loading}
              >
                {loading ? "Translating…" : "✨ Translate to ASL"}
              </button>
            </div>
          </div>

          {/* Gloss chips — shown after a successful translation */}
          {glosses.length > 0 && (
            <div className="translate-gloss">
              <span className="translate-gloss__label">ASL gloss:</span>
              <div className="translate-gloss__chips">
                {glosses.map((g, i) => (
                  <span key={i} className="translate-gloss__chip">{g}</span>
                ))}
              </div>
            </div>
          )}

          {/* <div className="translate-tip">
            💡 Tip: Keep sentences short for best results. The model works best with simple, clear phrases.
          </div> */}
        </div>

        {/* Output panel */}
        <div className="translate-output">
          {loading && (
            <div className="translate-state">
              <div className="spinner" />
              <p>Generating animation…</p>
            </div>
          )}
          {error && (
            <div className="translate-state">
              <p className="translate-error">⚠️ {error}</p>
            </div>
          )}
          {!loading && !error && frames.length > 0 && (
            <div style={{ textAlign: "center" }}>
              <SkeletonViewer frames={frames} fps={fps} autoPlay={true} />
              <p className="translate-animating">Animating: "{text}"</p>
            </div>
          )}
          {!loading && !error && frames.length === 0 && (
            <div className="translate-placeholder">
              <span className="placeholder-icon">🤲</span>
              <p>Your animation will appear here</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}