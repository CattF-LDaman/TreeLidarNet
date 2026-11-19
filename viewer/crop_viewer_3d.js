/**
 * Point cloud for one verifier crop — WebGPU when available, else WebGL2.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

function heightColor(z, zMin, zMax) {
  const t = zMax > zMin ? (z - zMin) / (zMax - zMin) : 0.5;
  const r = Math.round(52 + t * 200);
  const g = Math.round(211 - t * 80);
  const b = Math.round(153 + t * 40);
  return [r / 255, g / 255, b / 255];
}

function intensityColor(intensity) {
  const t = Math.min(1, Math.max(0, intensity));
  return [0.4 + t * 0.6, 0.35 + t * 0.45, 0.25 + t * 0.2];
}

async function createRenderer(canvas, status) {
  try {
    if (navigator.gpu) {
      const { WebGPURenderer } = await import("three/webgpu");
      const renderer = new WebGPURenderer({ canvas, antialias: true, alpha: true });
      await renderer.init();
      status("WebGPU");
      return { renderer, isWebGPU: true };
    }
  } catch (err) {
    console.warn("WebGPU unavailable", err);
    status("WebGPU failed — WebGL2");
  }
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  status("WebGL2");
  return { renderer, isWebGPU: false };
}

export async function mountCropViewer(canvas, { onStatus } = {}) {
  const status = onStatus || (() => {});

  const { renderer, isWebGPU } = await createRenderer(canvas, status);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 3;
  controls.maxDistance = 80;
  controls.screenSpacePanning = false;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minPolarAngle = 0.12;
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN,
  };
  controls.touches = {
    ONE: THREE.TOUCH.ROTATE,
    TWO: THREE.TOUCH.DOLLY_PAN,
  };

  let points = null;
  let footprint = null;
  const grid = new THREE.GridHelper(24, 12, 0x30363d, 0x21262d);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);
  const axis = new THREE.AxesHelper(2);
  scene.add(axis);

  function resize() {
    const w = canvas.clientWidth || 320;
    const h = canvas.clientHeight || 240;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  }

  async function renderFrame() {
    controls.update();
    if (isWebGPU && typeof renderer.renderAsync === "function") {
      await renderer.renderAsync(scene, camera);
    } else {
      renderer.render(scene, camera);
    }
  }

  function render() {
    renderFrame().catch((err) => console.error("render", err));
  }

  resize();
  const onResize = () => {
    resize();
    render();
  };
  window.addEventListener("resize", onResize);

  let raf = 0;
  function tick() {
    raf = requestAnimationFrame(tick);
    if (controls.enableDamping) {
      render();
    }
  }
  tick();

  function clearFootprint() {
    if (footprint) {
      scene.remove(footprint);
      footprint.geometry.dispose();
      footprint.material.dispose();
      footprint = null;
    }
  }

  function setCrop(payload) {
    if (points) {
      scene.remove(points);
      points.geometry.dispose();
      points.material.dispose();
      points = null;
    }
    clearFootprint();
    if (!payload || !payload.points || !payload.points.length) {
      render();
      return;
    }
    const pts = payload.points;
    const radius = payload.radius_m || 8;
    const ring = new THREE.RingGeometry(radius * 0.92, radius, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x58a6ff,
      transparent: true,
      opacity: 0.35,
      side: THREE.DoubleSide,
    });
    footprint = new THREE.Mesh(ring, ringMat);
    footprint.rotation.x = Math.PI / 2;
    scene.add(footprint);

    const positions = new Float32Array(pts.length * 3);
    const colors = new Float32Array(pts.length * 3);
    let zMin = Infinity;
    let zMax = -Infinity;
    for (const p of pts) {
      zMin = Math.min(zMin, p.z);
      zMax = Math.max(zMax, p.z);
    }
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
      let c;
      if (p.red != null && p.green != null && p.blue != null) {
        const s = (v) => (v > 1 ? v / 65535 : v);
        c = [s(p.red), s(p.green), s(p.blue)];
      } else {
        const useIntensity = false;
        c = useIntensity ? intensityColor((p.intensity || 0) / 65535) : heightColor(p.z, zMin, zMax);
      }
      colors[i * 3] = c[0];
      colors[i * 3 + 1] = c[1];
      colors[i * 3 + 2] = c[2];
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({
      size: 0.42,
      vertexColors: true,
      sizeAttenuation: true,
    });
    points = new THREE.Points(geom, mat);
    scene.add(points);

    geom.computeBoundingSphere();
    const bs = geom.boundingSphere;
    const r = Math.max(bs.radius, 2);
    const zSpan = zMax - zMin;
    const tz = zMin + zSpan * 0.45;
    const dist = Math.max(r * 2.4, 6);
    const az = Math.PI * 0.3;
    controls.target.set(0, 0, tz);
    camera.position.set(dist * Math.cos(az), dist * Math.sin(az), tz + r * 0.55);
    camera.up.set(0, 0, 1);
    controls.update();
    render();
  }

  function dispose() {
    cancelAnimationFrame(raf);
    window.removeEventListener("resize", onResize);
    clearFootprint();
    if (points) {
      points.geometry.dispose();
      points.material.dispose();
    }
    renderer.dispose();
  }

  return { setCrop, dispose, render, resize };
}