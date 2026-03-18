import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import SkeletonViewer from "../components/SkeletonViewer";
import stories from "../data/stories";
import { getStoryKeypoints } from "../services/api";
import "./StoryPlayer.css";

export default function StoryPlayer() {
  const { id } = useParams();
  const navigate = useNavigate();
  const story = stories.find((s) => s.id === Number(id));

  const [frames, setFrames]   = useState([]);   // ← was: keypoints
  const [fps, setFps]         = useState(25);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    if (!story) return;
    setLoading(true);
    setFrames([]);
    getStoryKeypoints(id)
      .then((data) => {
        // data.frames → [{joints: {nose:{x,y,z}, ...}}, ...]
        // data.fps    → 25
        setFrames(data.frames);
        setFps(data.fps);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [id, story]);

  if (!story) return (
    <div className="player-error">
      <h2>Story not found 😔</h2>
      <button className="btn-primary" onClick={() => navigate("/stories")}>← Back to stories</button>
    </div>
  );

  return (
    <main className="player-page">
      <button className="player-back" onClick={() => navigate("/stories")}>← Back</button>

      <div className="player-header">
        <div className="player-thumb" style={{ background: story.bg }}>
          {story.emoji}
        </div>
        <div>
          <span className="card-tag" style={{ background: story.tagBg }}>{story.tag}</span>
          <h1 className="player-title">{story.title}</h1>
          <p className="player-desc">{story.desc}</p>
          <span className="player-duration">⏱ {story.duration}</span>
        </div>
      </div>

      <div className="player-viewer-wrap">
        {loading && (
          <div className="player-state">
            <div className="spinner" />
            <p>Loading animation…</p>
          </div>
        )}
        {error && (
          <div className="player-state">
            <p className="player-error-msg">⚠️ {error}</p>
          </div>
        )}
        {!loading && !error && (
          // frames + fps wired directly into SkeletonViewer
          <SkeletonViewer frames={frames} fps={fps} autoPlay={true} />
        )}
      </div>
    </main>
  );
}