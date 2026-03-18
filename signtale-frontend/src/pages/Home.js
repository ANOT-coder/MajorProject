import { useNavigate } from "react-router-dom";
import StoryCard from "../components/StoryCard";
import SkeletonViewer from "../components/SkeletonViewer";
import stories from "../data/stories";
import "./Home.css";

export default function Home() {
  const navigate = useNavigate();

  return (
    <main>
      {/* ── Hero ── */}
      <section className="hero">
        <div className="hero-text">
          <div className="hero-badge">
            <span className="badge-dot" />Story Telling for the Hearing Impaired
          </div>
          <h1 className="hero-title">
            Stories that<br />
            <span className="highlight">speak</span> with<br />
            <em>their hands</em>
          </h1>
          <p className="hero-sub">
            Signtale brings stories to life through ASL animation — making every tale
            accessible, beautiful, and expressive.
          </p>
          <div className="hero-actions">
            <button className="btn-primary" onClick={() => navigate("/stories")}>
              Explore Stories →
            </button>
            <button className="btn-secondary" onClick={() => navigate("/translate")}>
              Try Translate
            </button>
          </div>
        </div>

        <div className="hero-visual">
          <span className="chip chip-1">👋 Hello!</span>
          <span className="chip chip-2">🤟 I love you</span>
          <span className="chip chip-3">✌️ Peace</span>
          <div className="hero-card-big">
            <SkeletonViewer />
            <p>Signing in progress…</p>
          </div>
        </div>
      </section>

      {/* ── Stats bar ── */}
      <div className="stats-bar">
        {[
          { num: "AI", label: "Story Translation" },
          { num: "3D", label: "Character Animation" },
          { num: "NLP", label: "ASL Gloss Conversion" },
          { num: "Web", label: "Interactive Interface" },
        ].map(({ num, label }) => (
          <div key={label} className="stat">
            <div className="stat-num">{num}</div>
            <div className="stat-label">{label}</div>
          </div>
        ))}
      </div>

      {/* ── Featured stories ── */}
      <section className="section">
        <div className="stories-header">
          <div>
            <span className="section-tag">📚 Featured</span>
            <h2 className="section-title">Stories to explore</h2>
          </div>
          <button className="btn-secondary home-view-all" onClick={() => navigate("/stories")}>
            View all →
          </button>
        </div>
        <div className="home-stories-grid">
          {stories.slice(0, 3).map((s) => (
            <StoryCard key={s.id} story={s} />
          ))}
        </div>
      </section>

      {/* ── Translate teaser ── */}
      <section className="section" style={{ paddingTop: 0 }}>
        <span className="section-tag">✏️ Try it live</span>
        <h2 className="section-title">Translate any text</h2>
        <p className="section-sub" style={{ marginBottom: "2rem" }}>
          Type any sentence and watch it transform into a beautiful ASL skeleton animation in real time.
        </p>
        <div className="translate-teaser">
          <div className="teaser-input-box">
            <label className="teaser-label">Your text</label>
            <div className="teaser-fake-input">Hello, how are you?</div>
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "1rem" }}>
              <button className="btn-primary" style={{ fontSize: "0.9rem", padding: "10px 22px" }}
                onClick={() => navigate("/translate")}>
                ✨ Try Translate
              </button>
            </div>
          </div>
          <div className="teaser-output">
            <SkeletonViewer />
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section className="section" style={{ paddingTop: 0 }}>
        <span className="section-tag">⚙️ Under the hood</span>
        <h2 className="section-title">How it works</h2>
        <div className="how-grid">
          {[
            { num: "1", icon: "📝", bg: "#FFF0A0", title: "You provide input", desc: "Type any text or pick a story from our growing library." },
            { num: "2", icon: "🧠", bg: "#D8CCFF", title: "ML generates keypoints", desc: "Our FastAPI backend runs the model and produces 3D joint data." },
            { num: "3", icon: "🎬", bg: "#B8F0D8", title: "Skeleton comes alive", desc: "Three.js renders a fluid, expressive ASL animation in your browser." },
          ].map((step) => (
            <div key={step.num} className="how-step">
              <span className="step-num">{step.num}</span>
              <div className="step-icon" style={{ background: step.bg }}>{step.icon}</div>
              <div className="step-title">{step.title}</div>
              <div className="step-desc">{step.desc}</div>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}