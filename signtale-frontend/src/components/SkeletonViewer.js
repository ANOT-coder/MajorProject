/**
 * SkeletonViewer.jsx  —  2D canvas + 3D Three.js skeleton renderer
 *
 * Props:
 *   frames    KeypointFrame[]  [{joints: {name: {x,y,z}}}]
 *   fps       number           playback speed (default 25)
 *   autoPlay  boolean          start on mount (default true)
 *
 * Toggle between 2D (canvas) and 3D (Three.js) with the button in the controls.
 *
 * 2D mode:
 *   - HTML5 Canvas, flat projection (x/y only, z ignored)
 *   - Glowing lines via shadowBlur — hands colour-coded
 *   - Head circle drawn around nose position
 *   - Cleaner for reading hand shapes
 *
 * 3D mode:
 *   - Three.js with orbit controls (drag / scroll / right-drag)
 *   - Cylinder bones, sphere joints, torus head ring
 */

import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";

// ─── Shared connectivity ──────────────────────────────────────────────────────
const HIDDEN_JOINTS = new Set([
  "left_knee", "right_knee", "left_ankle", "right_ankle",
  "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]);

const FACE_JOINTS = new Set([
  "left_eye_inner", "left_eye", "left_eye_outer",
  "right_eye_inner", "right_eye", "right_eye_outer",
  "left_ear", "right_ear", "mouth_left", "mouth_right",
]);

const BODY_CONNECTIONS = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_wrist", "left_pinky1"],
  ["left_wrist", "left_index1"],
  ["left_wrist", "left_thumb1"],
  ["right_wrist", "right_pinky1"],
  ["right_wrist", "right_index1"],
  ["right_wrist", "right_thumb1"],
  ["left_wrist", "left_hand_0"],
  ["right_wrist", "right_hand_0"],
];

const FINGER_CHAINS = [
  [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
];

function handConns(prefix) {
  return FINGER_CHAINS.flatMap(ch =>
    ch.slice(0, -1).map((_, i) => [`${prefix}_${ch[i]}`, `${prefix}_${ch[i + 1]}`])
  );
}

const ALL_CONNECTIONS = [
  ...BODY_CONNECTIONS,
  ...handConns("left_hand"),
  ...handConns("right_hand"),
];

// ─── Colours ──────────────────────────────────────────────────────────────────
const COL = {
  body: "#4a9eff",
  lh: "#9d6fff",
  rh: "#ff5f9e",
  bg: "#080c14",
};

function strokeColor(nameA) {
  if (nameA.startsWith("left_hand")) return COL.lh;
  if (nameA.startsWith("right_hand")) return COL.rh;
  return COL.body;
}
function dotColor(name) {
  if (name.startsWith("left_hand")) return COL.lh;
  if (name.startsWith("right_hand")) return COL.rh;
  return COL.body;
}

// ─── Shared: hand anchor fix ──────────────────────────────────────────────────
function anchorHandToWrist(joints, handPrefix, poseWristName) {
  const poseWrist = joints[poseWristName];
  const handRoot = joints[`${handPrefix}_0`];
  if (!poseWrist || !handRoot) return joints;
  const dx = poseWrist.x - handRoot.x;
  const dy = poseWrist.y - handRoot.y;
  const dz = poseWrist.z - handRoot.z;
  if (Math.abs(dx) < 0.005 && Math.abs(dy) < 0.005 && Math.abs(dz) < 0.005) return joints;
  const fixed = { ...joints };
  for (let i = 0; i <= 20; i++) {
    const key = `${handPrefix}_${i}`;
    if (fixed[key]) fixed[key] = { x: fixed[key].x + dx, y: fixed[key].y + dy, z: fixed[key].z + dz };
  }
  return fixed;
}

function prepJoints(rawJoints) {
  let j = rawJoints ?? {};
  j = anchorHandToWrist(j, "left_hand", "left_wrist");
  j = anchorHandToWrist(j, "right_hand", "right_wrist");
  return j;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2D CANVAS RENDERER
// ═════════════════════════════════════════════════════════════════════════════

// ── Replace the entire draw2D function ───────────────────────────────────

function draw2D(canvas, joints) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width / (window.devicePixelRatio || 1);
  const H = canvas.height / (window.devicePixelRatio || 1);

  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // ── Step 1: collect all visible joint positions ──────────────────────────
  const visible = Object.entries(joints).filter(
    ([name]) => !HIDDEN_JOINTS.has(name) && !FACE_JOINTS.has(name)
  );

  if (!visible.length) { ctx.restore(); return; }

  // ── Step 2: compute bounding box of all joints ───────────────────────────
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  for (const [, j] of visible) {
    if (j.x < minX) minX = j.x;
    if (j.x > maxX) maxX = j.x;
    if (j.y < minY) minY = j.y;
    if (j.y > maxY) maxY = j.y;
  }

  // ── Step 3: fit bounding box into canvas with padding ────────────────────
  const PAD = 0.04;                          // 12% padding on each side
  const bboxW = maxX - minX || 1;
  const bboxH = maxY - minY || 1;
  const scaleX = (W * (1 - PAD * 2)) / bboxW;
  const scaleY = (H * (1 - PAD * 2)) / bboxH;
  const scale  = Math.min(scaleX, scaleY);   // uniform scale — never distort

  // Centre the bounding box in the canvas
  const cx = W / 2 - ((minX + maxX) / 2) * scale;
  const cy = H / 2 - ((minY + maxY) / 2) * scale;

  const pt = j => ({ x: cx + j.x * scale, y: cy + j.y * scale });

  ctx.lineCap  = "round";
  ctx.lineJoin = "round";

  // ── Bones ──
  for (const [a, b] of ALL_CONNECTIONS) {
    const ja = joints[a], jb = joints[b];
    if (!ja || !jb) continue;
    const pa  = pt(ja), pb = pt(jb);
    const col = strokeColor(a);
    const isHand = a.startsWith("left_hand") || a.startsWith("right_hand");
    const aIdx   = isHand ? parseInt(a.split("_").pop(), 10) : -1;
    const isTip  = isHand && aIdx % 4 !== 0;

    ctx.lineWidth   = isHand ? (isTip ? 1 : 1.5) : 2;   // ← was 2/2.5/3.5
    ctx.strokeStyle = col;
    ctx.shadowColor = col;
    ctx.shadowBlur  = isHand ? 2 : 5;                    // ← was 5/10
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }

  ctx.shadowBlur = 0;

  // ── Body joint dots ──
  for (const [name, j] of Object.entries(joints)) {
    if (HIDDEN_JOINTS.has(name) || FACE_JOINTS.has(name)) continue;
    const p   = pt(j);
    const col = dotColor(name);
    const isHand = name.includes("_hand_");
    const r = isHand ? 1.5 :                                          
      ["left_shoulder","right_shoulder","left_hip","right_hip"].includes(name) ? 3.5 :   
      ["left_elbow","right_elbow","left_wrist","right_wrist"].includes(name)   ? 2.5 : 1.8; 
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();
  }

  // ── Face dots — small, subtle ──
  for (const [name, j] of Object.entries(joints)) {
    if (!FACE_JOINTS.has(name)) continue;
    const p = pt(j);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 1.8, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(120,180,255,0.45)";
    ctx.fill();
  }

  // ── Head circle around nose ──
  const nose = joints["nose"];
  if (nose) {
    const pn    = pt(nose);
    const headR = scale * 0.25;
    ctx.beginPath();
    ctx.arc(pn.x, pn.y, headR, 0, Math.PI * 2);
    ctx.strokeStyle = COL.body;
    ctx.lineWidth   = 2;
    ctx.shadowColor = COL.body;
    ctx.shadowBlur  = 8;
    ctx.stroke();
    ctx.shadowBlur  = 0;
  }

  ctx.restore();
}

function Viewer2D({ frames, frameIndex }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frames[frameIndex]) return;
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      const ctx = canvas.getContext("2d");
      ctx.scale(dpr, dpr);
    }
    draw2D(canvas, prepJoints(frames[frameIndex].joints));
  }, [frames, frameIndex]);

  return (
    <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// 3D THREE.JS RENDERER
// ═════════════════════════════════════════════════════════════════════════════

const C3 = {
  bodyBone: new THREE.Color(0x1a8fff),
  bodyJoint: new THREE.Color(0x55bbff),
  lhBone: new THREE.Color(0x8844ff),
  lhJoint: new THREE.Color(0xaa77ff),
  rhBone: new THREE.Color(0xff2288),
  rhJoint: new THREE.Color(0xff66bb),
};

function boneColor3(n) { return n.startsWith("left_hand") ? C3.lhBone : n.startsWith("right_hand") ? C3.rhBone : C3.bodyBone; }
function jointColor3(n) { return n.startsWith("left_hand") ? C3.lhJoint : n.startsWith("right_hand") ? C3.rhJoint : C3.bodyJoint; }

function jointRadius3(name) {
  if (name.includes("_hand_")) {
    const i = parseInt(name.split("_").pop(), 10);
    return [4, 8, 12, 16, 20].includes(i) ? 0.008 : (i === 0 ? 0.016 : 0.010);
  }
  if (["left_shoulder", "right_shoulder", "left_hip", "right_hip"].includes(name)) return 0.018;
  if (["left_elbow", "right_elbow"].includes(name)) return 0.014;
  if (["left_wrist", "right_wrist"].includes(name)) return 0.013;
  return 0.008;
}

function boneRadius3(a, b) {
  if (a.startsWith("left_hand") || a.startsWith("right_hand")) {
    const i = parseInt(a.split("_").pop(), 10);
    return i === 0 ? 0.014 : (i % 4 === 1 ? 0.011 : 0.008);
  }
  if (a.endsWith("shoulder") || b.endsWith("shoulder")) return 0.022;
  if (a.endsWith("elbow") || b.endsWith("elbow")) return 0.018;
  return 0.015;
}

const WS = 0.42;
function toWorld3(j) { return new THREE.Vector3(j.x * WS, -j.y * WS, j.z * WS); }

function makeBone3(pA, pB, color, r) {
  const dir = new THREE.Vector3().subVectors(pB, pA);
  const len = dir.length();
  if (len < 0.002) return null;
  const m = new THREE.Mesh(
    new THREE.CylinderGeometry(r, r * 0.85, len, 7),
    new THREE.MeshPhongMaterial({ color, shininess: 60 })
  );
  m.position.copy(pA).addScaledVector(dir, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
  return m;
}
function makeJoint3(pos, color, r) {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(r, 9, 9),
    new THREE.MeshPhongMaterial({ color, shininess: 90 })
  );
  m.position.copy(pos);
  return m;
}

function attachOrbit(camera, el) {
  let down = false, btn = -1, lx = 0, ly = 0;
  const sph = { theta: 0.25, phi: 1.15, radius: 4.2 };
  const tgt = new THREE.Vector3(0, 0.4, 0);
  const upd = () => {
    camera.position.set(
      tgt.x + sph.radius * Math.sin(sph.phi) * Math.sin(sph.theta),
      tgt.y + sph.radius * Math.cos(sph.phi),
      tgt.z + sph.radius * Math.sin(sph.phi) * Math.cos(sph.theta)
    );
    camera.lookAt(tgt);
  };
  upd();
  el.addEventListener("mousedown", e => { down = true; btn = e.button; lx = e.clientX; ly = e.clientY; e.preventDefault(); });
  window.addEventListener("mouseup", () => { down = false; });
  window.addEventListener("mousemove", e => {
    if (!down) return;
    const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY;
    if (btn === 0) { sph.theta -= dx * 0.007; sph.phi = Math.max(0.08, Math.min(Math.PI - 0.08, sph.phi + dy * 0.007)); }
    else if (btn === 2) { const r = new THREE.Vector3(); r.crossVectors(camera.getWorldDirection(new THREE.Vector3()), camera.up).normalize(); tgt.addScaledVector(r, -dx * 0.004); tgt.addScaledVector(camera.up, dy * 0.004); }
    upd();
  });
  el.addEventListener("wheel", e => { sph.radius = Math.max(1.5, Math.min(12, sph.radius + e.deltaY * 0.005)); upd(); e.preventDefault(); }, { passive: false });
  el.addEventListener("contextmenu", e => e.preventDefault());
}

function Viewer3D({ frames, frameIndex }) {
  const mountRef = useRef(null);
  const threeRef = useRef(null);

  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;
    const W = el.clientWidth || 420, H = el.clientHeight || 480;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H); renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor(0x080c14, 1); el.appendChild(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(48, W / H, 0.01, 100);
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x080c14, 0.12);
    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const kl = new THREE.DirectionalLight(0xffffff, 1.4); kl.position.set(2.5, 5, 3); scene.add(kl);
    const fl = new THREE.DirectionalLight(0x6699ff, 0.5); fl.position.set(-3, 1, -2); scene.add(fl);
    const rl = new THREE.DirectionalLight(0xaa66ff, 0.35); rl.position.set(0, -2, -4); scene.add(rl);
    const grid = new THREE.GridHelper(6, 20, 0x1a2540, 0x111a2e); grid.position.y = -1.5; scene.add(grid);
    const sg = new THREE.Group(); scene.add(sg);
    attachOrbit(camera, renderer.domElement);
    let id; const loop = () => { id = requestAnimationFrame(loop); renderer.render(scene, camera); }; loop();
    const ro = new ResizeObserver((entries) => {
      // requestAnimationFrame prevents the "undelivered notifications" warning
      // by deferring the resize handling to the next paint cycle
      window.requestAnimationFrame(() => {
        if (!entries.length) return;
        const w = el.clientWidth, h = el.clientHeight;
        if (!w || !h) return;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      });
    });
    ro.observe(el);
    threeRef.current = { scene, renderer, camera, sg };
    return () => { cancelAnimationFrame(id); ro.disconnect(); renderer.dispose(); if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement); threeRef.current = null; };
  }, []);

  useEffect(() => {
    const t = threeRef.current;
    if (!t || !frames[frameIndex]) return;
    const { sg } = t;
    while (sg.children.length) { const c = sg.children[0]; c.geometry?.dispose(); c.material?.dispose(); sg.remove(c); }

    const joints = prepJoints(frames[frameIndex].joints);

    for (const [name, j] of Object.entries(joints)) {
      if (HIDDEN_JOINTS.has(name) || FACE_JOINTS.has(name)) continue;
      sg.add(makeJoint3(toWorld3(j), jointColor3(name), jointRadius3(name)));
    }
    for (const [name, j] of Object.entries(joints)) {
      if (!FACE_JOINTS.has(name)) continue;
      sg.add(makeJoint3(toWorld3(j), new THREE.Color(0x4466aa), 0.006));
    }
    for (const [a, b] of ALL_CONNECTIONS) {
      const ja = joints[a], jb = joints[b];
      if (!ja || !jb) continue;
      const bone = makeBone3(toWorld3(ja), toWorld3(jb), boneColor3(a), boneRadius3(a, b));
      if (bone) sg.add(bone);
    }
    // Head torus
    const nj = joints["nose"];
    if (nj) {
      const torus = new THREE.Mesh(
        new THREE.TorusGeometry(0.20, 0.010, 8, 48),
        new THREE.MeshPhongMaterial({ color: new THREE.Color(0x7bb8ff), shininess: 50, transparent: true, opacity: 0.85 })
      );
      torus.position.copy(toWorld3(nj));
      sg.add(torus);
    }
  }, [frames, frameIndex]);

  return <div ref={mountRef} style={{ width: "100%", height: "100%", display: "block" }} />;
}

// ═════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═════════════════════════════════════════════════════════════════════════════
export default function SkeletonViewer({ frames = [], fps = 25, autoPlay = true, wordSegments = [], }) {
  const rafRef = useRef(null);
  const stateRef = useRef({ frameIndex: 0, lastTime: 0, playing: false });

  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [mode, setMode] = useState("2d");

  const msPerFrame = 1000 / fps;

  const tick = useCallback((now) => {
    const s = stateRef.current;
    if (!s.playing) return;
    if (now - s.lastTime >= msPerFrame) {
      s.lastTime = now;
      s.frameIndex = (s.frameIndex + 1) % Math.max(1, frames.length);
      setFrameIndex(s.frameIndex);
      setProgress(s.frameIndex / Math.max(1, frames.length - 1));
    }
    rafRef.current = requestAnimationFrame(tick);
  }, [frames.length, msPerFrame]);

  const play = useCallback(() => {
    if (!frames.length) return;
    stateRef.current.playing = true;
    stateRef.current.lastTime = 0;
    setPlaying(true);
    rafRef.current = requestAnimationFrame(tick);
  }, [frames.length, tick]);

  const pause = useCallback(() => {
    stateRef.current.playing = false;
    setPlaying(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  const restart = useCallback(() => {
    stateRef.current.frameIndex = 0;
    setFrameIndex(0); setProgress(0);
    if (stateRef.current.playing) { pause(); setTimeout(play, 40); }
  }, [pause, play]);

  const seek = useCallback((e) => {
    const pct = parseFloat(e.target.value);
    const idx = Math.round(pct * (frames.length - 1));
    stateRef.current.frameIndex = idx;
    setFrameIndex(idx); setProgress(pct);
  }, [frames.length]);

  useEffect(() => {
    if (!frames.length) return;
    stateRef.current.frameIndex = 0;
    setFrameIndex(0); setProgress(0);
    if (autoPlay) play();
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [frames]); // eslint-disable-line

  if (!frames.length) {
    return (
      <div style={S.empty}>
        <div style={S.emptyInner}>
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="10" r="4" stroke="#4a9eff" strokeWidth="2" />
            <line x1="24" y1="14" x2="24" y2="28" stroke="#4a9eff" strokeWidth="2" />
            <line x1="14" y1="18" x2="34" y2="18" stroke="#4a9eff" strokeWidth="2" />
            <line x1="24" y1="28" x2="16" y2="38" stroke="#4a9eff" strokeWidth="2" />
            <line x1="24" y1="28" x2="32" y2="38" stroke="#4a9eff" strokeWidth="2" />
          </svg>
          <p style={S.emptyText}>No animation data</p>
        </div>
      </div>
    );
  }

  const activeWord = wordSegments.find(
    seg => frameIndex >= seg.start_frame && frameIndex <= seg.end_frame
  )?.word ?? null;

  return (
    <div style={S.wrapper}>
      <div style={S.viewArea}>
        {mode === "2d"
          ? <Viewer2D frames={frames} frameIndex={frameIndex} />
          : <Viewer3D frames={frames} frameIndex={frameIndex} />
        }
      </div>

      <div style={S.legend}>
        <span style={{ ...S.dot, background: "#4a9eff" }} /> Body
        <span style={{ ...S.dot, background: "#9d6fff", marginLeft: 10 }} /> L Hand
        <span style={{ ...S.dot, background: "#ff5f9e", marginLeft: 10 }} /> R Hand
      </div>

      {mode === "3d" && <p style={S.hint}>⟳ Drag · Scroll zoom · Right-drag pan</p>}

      <div style={S.controls}>
        <button onClick={restart} style={S.btnSecondary} title="Restart">⏮</button>
        <button onClick={playing ? pause : play} style={S.btnPrimary} title={playing ? "Pause" : "Play"}>
          {playing ? "⏸" : "▶"}
        </button>
        <input type="range" min={0} max={1} step={0.001}
          value={progress} onChange={seek} style={S.scrubber} />
        <span style={S.counter}>
          {activeWord && (
            <span style={S.activeWord}>{activeWord}</span>
          )}
          {frameIndex + 1}
          <span style={{ opacity: 0.4 }}>/{frames.length}</span>
        </span>
        <button
          onClick={() => setMode(m => m === "2d" ? "3d" : "2d")}
          style={S.btnMode}
          title="Toggle 2D / 3D"
        >
          {mode === "2d" ? "3D" : "2D"}
        </button>
      </div>
    </div>
  );
}

const S = {
  wrapper: {
    position: "relative", display: "flex", flexDirection: "column",
    background: "#080c14", borderRadius: 14, overflow: "hidden",
    border: "1px solid #1e2d4a",
    boxShadow: "0 8px 40px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04)",
    fontFamily: "'DM Mono','Fira Mono','Consolas',monospace", userSelect: "none",
  },
  viewArea: {
    width: "100%", minHeight: 520, flex: 1, display: "block", background: "#080c14",
  },
  legend: {
    position: "absolute", top: 12, left: 14,
    display: "flex", alignItems: "center", gap: 4,
    fontSize: 11, color: "#7a9abf", letterSpacing: "0.04em", pointerEvents: "none",
  },
  dot: { display: "inline-block", width: 8, height: 8, borderRadius: "50%", marginRight: 4 },
  hint: {
    position: "absolute", top: 12, right: 14, margin: 0,
    fontSize: 10, color: "#334466", letterSpacing: "0.05em", pointerEvents: "none",
  },
  controls: {
    display: "flex", alignItems: "center", gap: 8, padding: "10px 14px",
    background: "rgba(8,12,20,0.9)", borderTop: "1px solid #1a2540", backdropFilter: "blur(6px)",
  },
  btnPrimary: {
    width: 36, height: 36, borderRadius: 8, border: "none",
    background: "linear-gradient(135deg,#2a5fff,#7b3fff)",
    color: "#fff", fontSize: 14, cursor: "pointer",
    display: "flex", alignItems: "center", justifyContent: "center",
    flexShrink: 0, boxShadow: "0 2px 10px rgba(100,80,255,0.4)",
  },
  btnSecondary: {
    width: 32, height: 32, borderRadius: 7, border: "1px solid #1e2d4a",
    background: "#0f1622", color: "#5577aa", fontSize: 13, cursor: "pointer",
    display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
  },
  btnMode: {
    height: 28, padding: "0 12px", borderRadius: 6,
    border: "1px solid #2a3d5a", background: "#0f1a2e",
    color: "#4a9eff", fontSize: 11, fontWeight: 600,
    cursor: "pointer", letterSpacing: "0.06em", flexShrink: 0,
  },
  scrubber: { flex: 1, height: 4, accentColor: "#4a9eff", cursor: "pointer" },
  counter: {
    fontSize: 11, color: "#4a9eff", minWidth: 52,
    textAlign: "right", letterSpacing: "0.04em", flexShrink: 0,
    display: "flex", flexDirection: "column", alignItems: "flex-end",  // ← add
  },
  empty: {
    display: "flex", alignItems: "center", justifyContent: "center",
    height: 300, background: "#080c14", borderRadius: 14, border: "1px solid #1e2d4a",
  },
  emptyInner: { display: "flex", flexDirection: "column", alignItems: "center", gap: 12 },
  emptyText: {
    margin: 0, fontSize: 13, color: "#334466",
    fontFamily: "'DM Mono',monospace", letterSpacing: "0.06em",
  },
  activeWord: {
    display: "block",
    fontSize: 10,
    color: "#ff5f9e",
    letterSpacing: "0.1em",
    fontWeight: 700,
    marginBottom: 1,
    textTransform: "uppercase",
  },

};