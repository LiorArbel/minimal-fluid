import * as THREE from 'three';
import Stats from 'stats-gl';
import { Fn, bool, float, globalId, texture, textureLoad, textureStore, uniform, vec2, vec4 } from 'three/tsl';
import { StorageTexture, WebGPURenderer } from 'three/webgpu';

const root = document.getElementById('app');
if (!root) {
  throw new Error('Missing #app');
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const renderer = new WebGPURenderer({ antialias: false, alpha: false });
renderer.setPixelRatio(1);
renderer.setSize(window.innerWidth, window.innerHeight);
const canvas = renderer.domElement;
if (!(canvas instanceof HTMLCanvasElement)) {
  throw new Error('WebGPURenderer returned OffscreenCanvas; expected HTMLCanvasElement for DOM usage');
}
root.appendChild(canvas);

let stats;
if (/stats=true/.test(window.location.toString())) {
  stats = new Stats({ trackGPU: true, trackCPT: true, logsPerSecond: 1, graphsPerSecond: 1 });
  stats.init(renderer);
  document.body.appendChild(stats.domElement);
}

const simScale = 1;
const simW = Math.max(8, Math.floor(window.innerWidth * simScale));
const simH = Math.max(8, Math.floor(window.innerHeight * simScale));
const simSize = vec2(simW, simH);
const substeps = 5;
const pressureLoops = 5;

const WG = [8, 8, 1];
const dispatchCount = [Math.ceil(simW / 8), Math.ceil(simH / 8)];

const pointer = { x: 0, y: 0, px: 0, py: 0, down: false };

const uniforms = {
  dt: uniform(float(1 / 60)),
  sdt: uniform(float(1 / (60 * substeps))),
  mouse: uniform(vec4(0, 0, 0, 0)),
  mouseToSim: uniform(vec2(simW / window.innerWidth, simH / window.innerHeight)),
  mouseDown: uniform(bool(false)),
  mouseRadius: uniform(float(Math.min(window.innerWidth, window.innerHeight) * 0.05)),
  omega: uniform(float(0.66)),
  pressureWarmstart: uniform(float(0.995))
};

function createStorageTexture(format) {
  const tex = new StorageTexture(simW, simH);
  tex.type = THREE.FloatType;
  tex.format = format;
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.colorSpace = THREE.NoColorSpace;
  tex.generateMipmaps = false;
  tex.wrapS = THREE.MirroredRepeatWrapping;
  tex.wrapT = THREE.MirroredRepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}

function clampCoord(c) {
  return vec2(c.x.clamp(0, simW - 1), c.y.clamp(0, simH - 1));
}

function uvFromPixel(p) {
  return p.add(0.5).div(simSize);
}

const velocityA = createStorageTexture(THREE.RGFormat);
const velocityB = createStorageTexture(THREE.RGFormat);
const smokeA = createStorageTexture(THREE.RedFormat);
const smokeB = createStorageTexture(THREE.RedFormat);
const pressureA = createStorageTexture(THREE.RedFormat);
const pressureB = createStorageTexture(THREE.RedFormat);
const divergence = createStorageTexture(THREE.RedFormat);

const advectVelocity = ({ src, dst }) => Fn(() => {
  const fragCoord = vec2(globalId.xy);
  const velocity = textureLoad(src, fragCoord).xy;
  const halfPrevPos = clampCoord(fragCoord.sub(velocity.mul(uniforms.sdt).mul(0.5)));
  const halfPrevVelocity = texture(src, uvFromPixel(halfPrevPos));
  const prevPos = clampCoord(fragCoord.sub(halfPrevVelocity.mul(uniforms.sdt)));
  const velocity_next = texture(src, uvFromPixel(prevPos)).xy;
  textureStore(dst, fragCoord, vec4(velocity_next, 0, 0));
})();

const advectSmoke = ({ velocitySrc, smokeSrc, smokeDst }) => Fn(() => {
  const fragCoord = vec2(globalId.xy);
  const velocity = textureLoad(velocitySrc, fragCoord).xy;
  const prevPos = fragCoord.sub(velocity.mul(uniforms.sdt));

  const smoke_advected = texture(smokeSrc, uvFromPixel(prevPos)).x;
  const mouseSim = vec2(
    uniforms.mouse.x.mul(uniforms.mouseToSim.x),
    uniforms.mouse.y.mul(uniforms.mouseToSim.y)
  );
  const mouseRadiusSim = uniforms.mouseRadius.mul(uniforms.mouseToSim.x);
  const mouseDist = fragCoord.distance(vec2(mouseSim.x, float(simH).sub(mouseSim.y))).length();
  const falloff = mouseDist.div(mouseRadiusSim).oneMinus().max(0).pow(2);
  const smoke_add = uniforms.mouseDown.select(uniforms.sdt.mul(10).mul(falloff), 0);
  const damping = uniforms.sdt.mul(0.1).oneMinus().max(0);

  const smoke_next = smoke_advected.add(smoke_add).mul(damping);
  textureStore(smokeDst, fragCoord, vec4(smoke_next, 0, 0, 0));
})();

const storeDivergence = ({ velocitySrc, divergenceDst }) => Fn(() => {
  const fragCoord = vec2(globalId.xy);
  const p_top = clampCoord(fragCoord.add(vec2(0, 1)));
  const p_bottom = clampCoord(fragCoord.add(vec2(0, -1)));
  const p_left = clampCoord(fragCoord.add(vec2(-1, 0)));
  const p_right = clampCoord(fragCoord.add(vec2(1, 0)));

  const v_top = textureLoad(velocitySrc, p_top).y;
  const v_bottom = textureLoad(velocitySrc, p_bottom).y;
  const v_left = textureLoad(velocitySrc, p_left).x;
  const v_right = textureLoad(velocitySrc, p_right).x;
  const v_div = v_right.sub(v_left).add(v_top.sub(v_bottom)).mul(0.5);

  textureStore(divergenceDst, fragCoord, vec4(v_div, 0, 0, 0));
})();

const jacobiPressure = ({ pressureSrc, pressureDst, divergenceSrc }) => Fn(() => {
  const fragCoord = vec2(globalId.xy);
  const p_center = textureLoad(pressureSrc, fragCoord).x;

  const p_top = textureLoad(pressureSrc, clampCoord(fragCoord.add(vec2(0, 1)))).x;
  const p_bottom = textureLoad(pressureSrc, clampCoord(fragCoord.add(vec2(0, -1)))).x;
  const p_left = textureLoad(pressureSrc, clampCoord(fragCoord.add(vec2(-1, 0)))).x;
  const p_right = textureLoad(pressureSrc, clampCoord(fragCoord.add(vec2(1, 0)))).x;

  const v_div = textureLoad(divergenceSrc, fragCoord).x;
  const p_jacobi = p_top.add(p_bottom).add(p_left).add(p_right).sub(v_div).mul(0.25);
  const p_next = p_center.mul(float(1).sub(uniforms.omega)).add(p_jacobi.mul(uniforms.omega));

  textureStore(pressureDst, fragCoord, vec4(p_next, 0, 0, 0));
})();

const projectAndForce = ({ velocitySrc, pressureSrc, pressureDst, velocityDst }) => Fn(() => {
  const fragCoord = vec2(globalId.xy);
  const velocity = textureLoad(velocitySrc, fragCoord).xy;

  const p = textureLoad(pressureSrc, fragCoord).x;
  const p_top = textureLoad(pressureSrc, clampCoord(fragCoord.add(vec2(0, 1)))).x;
  const p_bottom = textureLoad(pressureSrc, clampCoord(fragCoord.add(vec2(0, -1)))).x;
  const p_left = textureLoad(pressureSrc, clampCoord(fragCoord.add(vec2(-1, 0)))).x;
  const p_right = textureLoad(pressureSrc, clampCoord(fragCoord.add(vec2(1, 0)))).x;
  const p_grad = vec2(p_right.sub(p_left), p_top.sub(p_bottom)).mul(0.5);

  const v_projected = velocity.sub(p_grad);

  const mouseDelta = uniforms.mouse.xy.sub(uniforms.mouse.zw).mul(uniforms.mouseToSim);
  const mouseVel = mouseDelta.mul(2).mul(vec2(1, -1)).div(uniforms.sdt.max(1e-4));
  const mouseSim = vec2(
    uniforms.mouse.x.mul(uniforms.mouseToSim.x),
    uniforms.mouse.y.mul(uniforms.mouseToSim.y)
  );
  const mouseRadiusSim = uniforms.mouseRadius.mul(uniforms.mouseToSim.x);
  const mouseDist = fragCoord.distance(vec2(mouseSim.x, float(simH).sub(mouseSim.y))).length();
  const falloff = mouseDist.div(mouseRadiusSim).oneMinus().max(0).pow(2);
  const mouseForce = uniforms.mouseDown.select(mouseVel.mul(falloff), 0).mul(uniforms.sdt);
  const damping = uniforms.sdt.mul(0.1).oneMinus().max(0);

  const v_next = v_projected.mul(damping).add(mouseForce);
  textureStore(velocityDst, fragCoord, vec4(v_next, 0, 0));

  const warmStartP = p.mul(uniforms.pressureWarmstart);
  textureStore(pressureDst, fragCoord, vec4(warmStartP, 0, 0, 0));
})();

const copyTexture = ({ src, dst }) => Fn(() => {
  const p = vec2(globalId.xy);
  textureStore(dst, p, textureLoad(src, p));
})();

const advectVelocityCompute = advectVelocity({ src: velocityA, dst: velocityB }).computeKernel(WG);
const advectSmokeCompute = advectSmoke({ velocitySrc: velocityB, smokeSrc: smokeA, smokeDst: smokeB }).computeKernel(WG);
const divergenceCompute = storeDivergence({ velocitySrc: velocityB, divergenceDst: divergence }).computeKernel(WG);
const pressureABCompute = jacobiPressure({ pressureSrc: pressureA, pressureDst: pressureB, divergenceSrc: divergence }).computeKernel(WG);
const pressureBACompute = jacobiPressure({ pressureSrc: pressureB, pressureDst: pressureA, divergenceSrc: divergence }).computeKernel(WG);
const projectForceCompute = projectAndForce({
  velocitySrc: velocityB,
  pressureSrc: pressureA,
  pressureDst: pressureB,
  velocityDst: velocityA
}).computeKernel(WG);

const copySmokeBToA = copyTexture({ src: smokeB, dst: smokeA }).computeKernel(WG);
const copyPressureBToA = copyTexture({ src: pressureB, dst: pressureA }).computeKernel(WG);

const pressurePass = [...Array(pressureLoops)].flatMap(() => [pressureABCompute, pressureBACompute]);
const substepPass = [
  advectVelocityCompute,
  advectSmokeCompute,
  divergenceCompute,
  ...pressurePass,
  projectForceCompute,
  copySmokeBToA,
  copyPressureBToA
];
const framePass = [...Array(substeps)].flatMap(() => substepPass);

const material = new THREE.MeshBasicMaterial({ transparent: true });
material.colorNode = Fn(() => {
  const v = texture(velocityA).xy;
  const s = texture(smokeA).x;
  const speed = v.length();
  const maxSpeed = float(0.5).div(uniforms.sdt.max(1e-4));

  const r = speed.div(maxSpeed).mul(0.5).oneMinus();
  const g = speed.div(maxSpeed);
  const b = speed.div(maxSpeed).oneMinus();

  return vec4(b, g, r, s);
})();

const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
scene.add(quad);

const updateMouseFromEvent = (e) => {
  const rect = canvas.getBoundingClientRect();
  pointer.x = e.clientX - rect.left;
  pointer.y = e.clientY - rect.top;
};

canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  updateMouseFromEvent(e);
  pointer.px = pointer.x;
  pointer.py = pointer.y;
  pointer.down = true;
});

canvas.addEventListener('pointerup', (e) => {
  canvas.releasePointerCapture(e.pointerId);
  pointer.down = false;
});

canvas.addEventListener('pointermove', (e) => {
  updateMouseFromEvent(e);
});

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
});

let lastTime = 0;
let lastTimestamps = 0;
renderer.init().then(() => {
  renderer.setAnimationLoop((timeMs) => {
    const dt = Math.min(Math.max((timeMs - lastTime) * 0.001, 1 / 240), 1 / 30);
    lastTime = timeMs;

    uniforms.dt.value = dt;
    uniforms.sdt.value = dt / substeps;
    uniforms.mouseToSim.value.set(simW / canvas.clientWidth, simH / canvas.clientHeight);
    uniforms.mouse.value.set(pointer.x, pointer.y, pointer.px, pointer.py);
    uniforms.mouseDown.value = pointer.down;

    renderer.compute(framePass, dispatchCount);

    pointer.px = pointer.x;
    pointer.py = pointer.y;

    renderer.render(scene, camera);
    if (stats && timeMs - lastTimestamps > 1000) {
      renderer.resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE);
      renderer.resolveTimestampsAsync(THREE.TimestampQuery.RENDER);
      lastTimestamps = timeMs;
    }
    stats?.update();
  });
});
