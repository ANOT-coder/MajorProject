/**
 * SkeletonViewer.jsx  —  Three.js 3D skeleton renderer
 *
 * Props (identical to old canvas version — no changes needed in Translate/StoryPlayer):
 *   frames    KeypointFrame[]  [{joints: {name: {x,y,z}}}]  x/y/z in [0,1]
 *   fps       number           playback speed (default 25)
 *   autoPlay  boolean          start on mount (default true)
 *
 * Features:
 *   • Real 3D — orbit with left-drag, zoom with scroll, pan with right-drag
 *   • Spheres for joints, CylinderGeometry bones between them
 *   • Colour-coded: body=blue, left hand=purple, right hand=pink
 *   • Play/Pause, Restart, scrubber — same controls as before
 *   • Fully self-contained — no extra npm packages beyond three (already in vite)
 *
 * Install if not already present:
 *   npm install three
 */

import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import "./SkeletonViewer.css";

// ─── Skeleton connectivity ────────────────────────────────────────────────────
// Connections use exact JOINT_NAMES from inference.py (MediaPipe Pose order)
const BODY_CONNECTIONS = [
  // Torso
  ["left_shoulder",  "right_shoulder"],
  ["left_shoulder",  "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip",       "right_hip"],
  // Arms
  ["left_shoulder",  "left_elbow"],
  ["left_elbow",     "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow",    "right_wrist"],
  // Wrist → body finger tips (pose landmarks 17-22)
  ["left_wrist",     "left_pinky1"],
  ["left_wrist",     "left_index1"],
  ["left_wrist",     "left_thumb1"],
  ["right_wrist",    "right_pinky1"],
  ["right_wrist",    "right_index1"],
  ["right_wrist",    "right_thumb1"],
  // Legs
  ["left_hip",       "left_knee"],
  ["right_hip",      "right_knee"],
  ["left_knee",      "left_ankle"],
  ["right_knee",     "right_ankle"],
  ["left_ankle",     "left_heel"],
  ["right_ankle",    "right_heel"],
  ["left_ankle",     "left_foot_index"],
  ["right_ankle",    "right_foot_index"],
  // Wrist to hand_0 (connects pose wrist → hand root)
  ["left_wrist",     "left_hand_0"],
  ["right_wrist",    "right_hand_0"],
];

const FINGER_CHAINS = [
  [0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20],
];

function handConns(prefix) {
  return FINGER_CHAINS.flatMap(ch =>
    ch.slice(0,-1).map((_,i) => [`${prefix}_${ch[i]}`, `${prefix}_${ch[i+1]}`])
  );
}

const ALL_CONNECTIONS = [
  ...BODY_CONNECTIONS,
  ...handConns("left_hand"),
  ...handConns("right_hand"),
];

// ─── Colours ──────────────────────────────────────────────────────────────────
const COLOR_BODY       = new THREE.Color(0x3b82f6);  // blue
const COLOR_LEFT_HAND  = new THREE.Color(0x8b5cf6);  // purple
const COLOR_RIGHT_HAND = new THREE.Color(0xec4899);  // pink
const COLOR_JOINT_BODY = new THREE.Color(0x60a5fa);
const COLOR_JOINT_LEFT = new THREE.Color(0xa78bfa);
const COLOR_JOINT_RIGHT= new THREE.Color(0xf472b6);

function boneColor(nameA) {
  if (nameA.startsWith("left_hand"))  return COLOR_LEFT_HAND;
  if (nameA.startsWith("right_hand")) return COLOR_RIGHT_HAND;
  return COLOR_BODY;
}
function jointColor(name) {
  if (name.startsWith("left_hand"))  return COLOR_JOINT_LEFT;
  if (name.startsWith("right_hand")) return COLOR_JOINT_RIGHT;
  return COLOR_JOINT_BODY;
}

// ─── Convert training-space coords → Three.js world coords ───────────────────
// Model outputs anchor/scale normalized values (centered at 0, ~[-3,3] range).
// preprocess_to_npz_v3: anchor=mid-hip, scale=shoulder-width, y increases DOWN.
// Three.js: y increases UP — so we flip y.
// We scale down by ~0.5 so the figure fits the viewport comfortably.
const WORLD_SCALE = 0.5;
function toWorld(j) {
  return new THREE.Vector3(
     j.x * WORLD_SCALE,   // x: left/right as-is
    -j.y * WORLD_SCALE,   // y: flip (image coords → Three.js coords)
     j.z * WORLD_SCALE,   // z: depth as-is
  );
}

// ─── Build a bone (cylinder) between two points ───────────────────────────────
function makeBone(posA, posB, color, isHand) {
  const dir    = new THREE.Vector3().subVectors(posB, posA);
  const length = dir.length();
  if (length < 0.001) return null;

  const radius   = isHand ? 0.012 : 0.022;
  const geo      = new THREE.CylinderGeometry(radius, radius, length, 6);
  const mat      = new THREE.MeshPhongMaterial({ color, shininess: 40 });
  const mesh     = new THREE.Mesh(geo, mat);

  // Position at midpoint, orient along direction
  mesh.position.copy(posA).addScaledVector(dir, 0.5);
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.normalize()
  );
  return mesh;
}

// ─── Build a joint sphere ─────────────────────────────────────────────────────
function makeJoint(pos, color, isHand) {
  const radius = isHand ? 0.025 : 0.040;
  const geo    = new THREE.SphereGeometry(radius, 8, 8);
  const mat    = new THREE.MeshPhongMaterial({ color, shininess: 80 });
  const mesh   = new THREE.Mesh(geo, mat);
  mesh.position.copy(pos);
  return mesh;
}

// ─── OrbitControls (inline, no import needed) ─────────────────────────────────
// Minimal orbit: left-drag=rotate, right-drag=pan, scroll=zoom
function attachOrbit(camera, domElement) {
  let isDown = false, button = -1;
  let lastX = 0, lastY = 0;
  let spherical = { theta: 0.3, phi: 1.2, radius: 4 };
  let target = new THREE.Vector3(0, 0, 0);

  function updateCamera() {
    camera.position.set(
      target.x + spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta),
      target.y + spherical.radius * Math.cos(spherical.phi),
      target.z + spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta),
    );
    camera.lookAt(target);
  }
  updateCamera();

  domElement.addEventListener("mousedown", (e) => {
    isDown = true; button = e.button;
    lastX = e.clientX; lastY = e.clientY;
    e.preventDefault();
  });
  window.addEventListener("mouseup",   () => { isDown = false; });
  window.addEventListener("mousemove", (e) => {
    if (!isDown) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;

    if (button === 0) {
      // rotate
      spherical.theta -= dx * 0.008;
      spherical.phi    = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi + dy * 0.008));
    } else if (button === 2) {
      // pan
      const right = new THREE.Vector3();
      const up    = new THREE.Vector3();
      camera.getWorldDirection(new THREE.Vector3());
      right.crossVectors(camera.getWorldDirection(new THREE.Vector3()), camera.up).normalize();
      up.copy(camera.up).normalize();
      target.addScaledVector(right, -dx * 0.004);
      target.addScaledVector(up,     dy * 0.004);
    }
    updateCamera();
  });
  domElement.addEventListener("wheel", (e) => {
    spherical.radius = Math.max(1, Math.min(10, spherical.radius + e.deltaY * 0.005));
    updateCamera();
    e.preventDefault();
  }, { passive: false });
  domElement.addEventListener("contextmenu", (e) => e.preventDefault());
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function SkeletonViewer({
  frames   = [],
  fps      = 25,
  autoPlay = true,
}) {
  const mountRef  = useRef(null);
  const threeRef  = useRef(null);   // { scene, renderer, camera, skeletonGroup }
  const rafRef    = useRef(null);
  const stateRef  = useRef({ frameIndex: 0, lastTime: 0, playing: false });

  const [frameIndex, setFrameIndex] = useState(0);
  const [playing,    setPlaying]    = useState(false);
  const [progress,   setProgress]   = useState(0);

  const msPerFrame = 1000 / fps;

  // ── Init Three.js scene once ────────────────────────────────────────────────
  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const W = el.clientWidth  || 400;
    const H = el.clientHeight || 500;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x0d1117, 1);
    el.appendChild(renderer.domElement);

    // Camera
    const camera = new THREE.PerspectiveCamera(50, W / H, 0.01, 100);
    camera.position.set(0, 0, 4);

    // Scene
    const scene = new THREE.Scene();

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(2, 4, 3);
    scene.add(dirLight);
    const fillLight = new THREE.DirectionalLight(0x8080ff, 0.3);
    fillLight.position.set(-2, -1, -2);
    scene.add(fillLight);

    // Skeleton group (rebuilt each frame)
    const skeletonGroup = new THREE.Group();
    scene.add(skeletonGroup);

    // Orbit controls
    attachOrbit(camera, renderer.domElement);

    // Render loop (continuous — needed for orbit to feel live)
    let animId;
    function renderLoop() {
      animId = requestAnimationFrame(renderLoop);
      renderer.render(scene, camera);
    }
    renderLoop();

    // Resize observer
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(el);

    threeRef.current = { scene, renderer, camera, skeletonGroup };

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      threeRef.current = null;
    };
  }, []); // eslint-disable-line

  // ── Rebuild skeleton meshes for a given frame ───────────────────────────────
  const renderFrame = useCallback((idx) => {
    const t = threeRef.current;
    if (!t || !frames[idx]) return;
    const { skeletonGroup } = t;
    const joints = frames[idx].joints ?? {};

    // Clear old meshes
    while (skeletonGroup.children.length) {
      const child = skeletonGroup.children[0];
      child.geometry?.dispose();
      child.material?.dispose();
      skeletonGroup.remove(child);
    }

    // Add joint spheres
    for (const [name, j] of Object.entries(joints)) {
      const pos  = toWorld(j);
      const isHand = name.includes("hand");
      const mesh = makeJoint(pos, jointColor(name), isHand);
      skeletonGroup.add(mesh);
    }

    // Add bone cylinders
    for (const [a, b] of ALL_CONNECTIONS) {
      const ja = joints[a], jb = joints[b];
      if (!ja || !jb) continue;
      const posA   = toWorld(ja);
      const posB   = toWorld(jb);
      const isHand = a.includes("hand");
      const bone   = makeBone(posA, posB, boneColor(a), isHand);
      if (bone) skeletonGroup.add(bone);
    }
  }, [frames]);

  // ── Animation tick ──────────────────────────────────────────────────────────
  const tick = useCallback((now) => {
    const s = stateRef.current;
    if (!s.playing) return;
    if (now - s.lastTime >= msPerFrame) {
      s.lastTime   = now;
      s.frameIndex = (s.frameIndex + 1) % Math.max(1, frames.length);
      setFrameIndex(s.frameIndex);
      setProgress(s.frameIndex / Math.max(1, frames.length - 1));
      renderFrame(s.frameIndex);
    }
    rafRef.current = requestAnimationFrame(tick);
  }, [frames.length, msPerFrame, renderFrame]);

  const play = useCallback(() => {
    if (!frames.length) return;
    stateRef.current.playing  = true;
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
    renderFrame(0);
    if (stateRef.current.playing) { pause(); setTimeout(play, 40); }
  }, [pause, play, renderFrame]);

  const seek = useCallback((e) => {
    const pct = parseFloat(e.target.value);
    const idx = Math.round(pct * (frames.length - 1));
    stateRef.current.frameIndex = idx;
    setFrameIndex(idx); setProgress(pct);
    renderFrame(idx);
  }, [frames.length, renderFrame]);

  // ── When frames arrive, render first frame + autoplay ───────────────────────
  useEffect(() => {
    if (!frames.length) return;
    stateRef.current.frameIndex = 0;
    setFrameIndex(0); setProgress(0);
    renderFrame(0);
    if (autoPlay) play();
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [frames]); // eslint-disable-line

  // ── Render ───────────────────────────────────────────────────────────────────
  if (!frames.length) {
    return (
      <div className="skeleton-viewer skeleton-viewer--empty">
        <p>No animation data</p>
      </div>
    );
  }

  return (
    <div className="skeleton-viewer">
      {/* Three.js mounts into this div */}
      <div ref={mountRef} className="skeleton-viewer__canvas" />

      <p className="skeleton-viewer__hint">🖱 Drag to orbit · Scroll to zoom · Right-drag to pan</p>

      <div className="skeleton-viewer__controls">
        <button onClick={restart} className="sv-btn" title="Restart">⏮</button>
        <button
          onClick={playing ? pause : play}
          className="sv-btn sv-btn--primary"
          title={playing ? "Pause" : "Play"}
        >
          {playing ? "⏸" : "▶"}
        </button>
        <input
          type="range" min={0} max={1} step={0.001}
          value={progress} onChange={seek}
          className="sv-scrubber"
        />
        <span className="sv-counter">{frameIndex + 1} / {frames.length}</span>
      </div>
    </div>
  );
}