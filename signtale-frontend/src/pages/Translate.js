import { useState } from "react";
import SkeletonViewer from "../components/SkeletonViewer";
import { translateText } from "../services/api";
import "./Translate.css";

export default function Translate() {
  const [text, setText]       = useState("");
  const [frames, setFrames]   = useState([]);
  const [fps, setFps]         = useState(25);
  const [glosses, setGlosses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [wordSegments, setWordSegments] = useState([]);


  const handleTranslate = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setFrames([]);
    setGlosses([]);
    setWordSegments([]); 
    try {
      const data = await translateText(text);
      setFrames(data.frames);
      setFps(data.fps);
      setGlosses(data.glosses);
      setWordSegments(data.word_segments ?? []);
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

      <div className="translate-layout">

        {/* ── Left: input panel ── */}
        <div className="translate-input-panel">
          <div className="translate-textarea-wrap">
            <label className="translate-label">Your text</label>
            <textarea
              className="translate-textarea"
              placeholder="e.g. I am happy to meet you"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleTranslate()}
            />
            <span className="translate-charcount">{text.length} chars</span>
          </div>

          <div className="translate-actions">
            <button
              className="btn-translate"
              onClick={handleTranslate}
              disabled={!text.trim() || loading}
            >
              {loading ? "Translating…" : "✨ Translate to ASL"}
            </button>
          </div>

          {error && (
            <div className="translate-error">
              ⚠️ {error}
            </div>
          )}

          {/* Gloss chips */}
          {glosses.length > 0 && (
            <div className="translate-gloss-panel">
              <div className="translate-gloss-body">
                <div className="translate-gloss-chips">
                  {glosses.map((g, i) => (
                    <span key={i} className="translate-gloss-chip">{g}</span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── Right: viewer panel ── */}
        <div className="translate-viewer-panel">
          <div className="translate-viewer-card" style={{ minHeight: "600px" }}>

            {/* Header bar */}
            <div className="translate-viewer-header">
              <div className="translate-viewer-title">
                <span>ASL · Live Preview</span>
              </div>
              <div className="translate-viewer-dots">
                <span className="translate-viewer-dot translate-viewer-dot--r" />
                <span className="translate-viewer-dot translate-viewer-dot--y" />
                <span className={`translate-viewer-dot translate-viewer-dot--g ${frames.length === 0 ? "idle" : ""}`} />
              </div>
            </div>

            {/* Body — skeleton or state */}
            <div className="translate-viewer-body">
              {loading && (
                <div className="translate-viewer-loading">
                  <div className="translate-viewer-loading-bar">
                    <div className="translate-viewer-loading-fill" />
                  </div>
                  <span>Generating animation…</span>
                </div>
              )}

              {!loading && frames.length > 0 && (
                <SkeletonViewer frames={frames} fps={fps} autoPlay={true} wordSegments={wordSegments} />
              )}

              {!loading && frames.length === 0 && !error && (
                <div className="translate-viewer-placeholder">
                  <span className="translate-placeholder-icon"></span>
                  <span className="translate-placeholder-label">Awaiting input</span>
                  <span className="translate-placeholder-sub">
                    Enter text and click translate
                  </span>
                </div>
              )}

              {!loading && error && (
                <div className="translate-viewer-placeholder">
                  <span className="translate-placeholder-icon">⚠️</span>
                  <span className="translate-placeholder-label">Generation failed</span>
                  <span className="translate-placeholder-sub">{error}</span>
                </div>
              )}
            </div>

            {/* Caption strip */}
            <div className="translate-viewer-caption">
              <span className="translate-viewer-caption-text">
                {frames.length > 0 ? `▶ ${text}` : "—"}
              </span>
              <span className="translate-viewer-caption-fps">
                {frames.length > 0 ? `${fps} fps · ${frames.length} frames` : ""}
              </span>
            </div>

          </div>
        </div>

      </div>
    </main>
  );
}