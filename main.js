import * as THREE from 'three';
import Stats from 'stats-gl';
import GUI from 'lil-gui';
import WebGPU from 'three/addons/capabilities/WebGPU.js';
import {
  Fn,
  If,
  Return,
  atomicAdd,
  atomicLoad,
  atomicStore,
  bool,
  float,
  globalId,
  instanceIndex,
  instancedArray,
  int,
  mix,
  positionGeometry,
  texture,
  textureLoad,
  textureStore,
  uniform,
  vec2,
  vec3,
  vec4
} from 'three/tsl';
import { StorageTexture, WebGPURenderer } from 'three/webgpu';

const root = document.getElementById('stage');
if (!root) {
  throw new Error('Missing #stage');
}
if (WebGPU.isAvailable() === false) {
  document.body.appendChild(WebGPU.getErrorMessage());
  throw new Error('WebGPU is not available');
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
  stats.domElement.id = 'stats';
  document.body.appendChild(stats.domElement);
}
const debugPressure = /debugPressure=true/.test(window.location.toString());

const simScale = 0.25;
const simW = Math.max(8, Math.floor(window.innerWidth * simScale));
const simH = Math.max(8, Math.floor(window.innerHeight * simScale));
const simSize = vec2(simW, simH);

const uW = simW + 1;
const uH = simH;
const vW = simW;
const vH = simH + 1;

const blobW = Math.max(12, Math.floor(simW * 0.45));
const blobH = Math.max(12, Math.floor(simH * 0.3));
const blobX = Math.floor((simW - blobW) * 0.5);
const blobY = Math.floor(simH * 0.15);
const particlesPerCell = 4;
const particleAmount = blobW * blobH * particlesPerCell;

const substeps = 5;
const WG = [8, 8, 1];
const cellDispatch = [Math.ceil(simW / 8), Math.ceil(simH / 8)];
const uDispatch = [Math.ceil(uW / 8), Math.ceil(uH / 8)];
const vDispatch = [Math.ceil(vW / 8), Math.ceil(vH / 8)];
const atomicScale = 4096;
const liquidThreshold = 0.25;

const pointer = { x: 0, y: 0, px: 0, py: 0, down: false };
const gravity = vec2(0, simH);
const dampingRate = 0.1;
const params = {
  pressureLoops: 20,
  underrelaxation: 0.66,
  interactionRadius: Math.min(window.innerWidth, window.innerHeight) * 0.05,
  flipAmount: 0.95
};

const uniforms = {
  sdt: uniform(float(1 / (60 * substeps))),
  mouse: uniform(vec4(0, 0, 0, 0)),
  mouseToSim: uniform(vec2(simW / window.innerWidth, simH / window.innerHeight)),
  mouseDown: uniform(bool(false)),
  interactionRadius: uniform(float(params.interactionRadius)),
  underrelaxation: uniform(float(params.underrelaxation)),
  flipAmount: uniform(float(params.flipAmount))
};

const gui = new GUI();
gui.close();
gui.domElement.style.top = '0';
gui.domElement.style.bottom = 'auto';
gui.domElement.style.right = '0';
gui.add(params, 'pressureLoops', 1, 80, 1).onFinishChange((value) => {
  params.pressureLoops = Math.round(value);
});
gui.add(params, 'underrelaxation', 0, 1, 0.001).onChange((value) => {
  uniforms.underrelaxation.value = value;
});
gui.add(params, 'interactionRadius', 1, Math.min(window.innerWidth, window.innerHeight) * 0.5, 1).onChange((value) => {
  uniforms.interactionRadius.value = value;
});
gui.add(params, 'flipAmount', 0, 1, 0.001).onChange((value) => {
  uniforms.flipAmount.value = value;
});

function createStorageTexture(width, height, format) {
  const tex = new StorageTexture(width, height);
  tex.type = THREE.FloatType;
  tex.format = format;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

function clampCellCoord(c) {
  return vec2(c.x.clamp(0, simW - 1), c.y.clamp(0, simH - 1));
}

function clampUCoord(c) {
  return vec2(c.x.clamp(0, uW - 1), c.y.clamp(0, uH - 1));
}

function clampVCoord(c) {
  return vec2(c.x.clamp(0, vW - 1), c.y.clamp(0, vH - 1));
}

function clampParticlePos(p) {
  return vec2(p.x.clamp(0.5, simW - 0.5), p.y.clamp(0.5, simH - 0.5));
}

function guardBounds(p, w, h) {
  If(p.x.greaterThanEqual(w).or(p.y.greaterThanEqual(h)), () => {
    Return();
  });
}

function cellIndex(c) {
  return c.x.add(c.y.mul(simW));
}

function uIndex(c) {
  return c.x.add(c.y.mul(uW));
}

function vIndex(c) {
  return c.x.add(c.y.mul(vW));
}

function cellCoordFromPos(p) {
  return clampCellCoord(p.sub(0.5).floor());
}

function uUVFromPos(p) {
  return vec2(p.x.add(0.5).div(uW), p.y.div(uH));
}

function vUVFromPos(p) {
  return vec2(p.x.div(vW), p.y.add(0.5).div(vH));
}

function sampleVelocity(uTex, vTex, p) {
  return vec2(texture(uTex, uUVFromPos(p)).x, texture(vTex, vUVFromPos(p)).x);
}

function uFaceMask(p) {
  const leftAvail = p.x.min(1);
  const rightAvail = float(uW - 1).sub(p.x).min(1);
  const interior = leftAvail.mul(rightAvail);
  const leftCell = clampCellCoord(vec2(p.x.sub(1), p.y));
  const rightCell = clampCellCoord(vec2(p.x, p.y));
  const leftMask = textureLoad(liquidMask, leftCell).x.mul(leftAvail);
  const rightMask = textureLoad(liquidMask, rightCell).x.mul(rightAvail);
  return leftMask.max(rightMask).mul(interior);
}

function vFaceMask(p) {
  const bottomAvail = p.y.min(1);
  const topAvail = float(vH - 1).sub(p.y).min(1);
  const interior = bottomAvail.mul(topAvail);
  const bottomCell = clampCellCoord(vec2(p.x, p.y.sub(1)));
  const topCell = clampCellCoord(vec2(p.x, p.y));
  const bottomMask = textureLoad(liquidMask, bottomCell).x.mul(bottomAvail);
  const topMask = textureLoad(liquidMask, topCell).x.mul(topAvail);
  return bottomMask.max(topMask).mul(interior);
}

const uA = createStorageTexture(uW, uH, THREE.RedFormat);
const uB = createStorageTexture(uW, uH, THREE.RedFormat);
const uC = createStorageTexture(uW, uH, THREE.RedFormat);
const vA = createStorageTexture(vW, vH, THREE.RedFormat);
const vB = createStorageTexture(vW, vH, THREE.RedFormat);
const vC = createStorageTexture(vW, vH, THREE.RedFormat);
const uValidA = createStorageTexture(uW, uH, THREE.RedFormat);
const uValidB = createStorageTexture(uW, uH, THREE.RedFormat);
const vValidA = createStorageTexture(vW, vH, THREE.RedFormat);
const vValidB = createStorageTexture(vW, vH, THREE.RedFormat);
const pressureA = createStorageTexture(simW, simH, THREE.RedFormat);
const pressureB = createStorageTexture(simW, simH, THREE.RedFormat);
const divergence = createStorageTexture(simW, simH, THREE.RedFormat);
const projectedDivergence = createStorageTexture(simW, simH, THREE.RedFormat);
const liquidMask = createStorageTexture(simW, simH, THREE.RedFormat);
const displayVelocity = createStorageTexture(simW, simH, THREE.RGFormat);

const particlePositions = instancedArray(particleAmount, 'vec2');
const particleVelocities = instancedArray(particleAmount, 'vec2');
const uTemp = instancedArray(uW * uH, 'int').toAtomic();
const uWeightTemp = instancedArray(uW * uH, 'int').toAtomic();
const vTemp = instancedArray(vW * vH, 'int').toAtomic();
const vWeightTemp = instancedArray(vW * vH, 'int').toAtomic();
const cellTemp = instancedArray(simW * simH, 'int').toAtomic();

const scatterParticles = Fn(() => {
  const position = particlePositions.element(instanceIndex);
  const velocity = particleVelocities.element(instanceIndex);
  const particleIndex = float(instanceIndex);
  const cellIndexInBlob = particleIndex.div(particlesPerCell).floor();
  const patternY = particleIndex.sub(cellIndexInBlob.mul(particlesPerCell)).div(2).floor();
  const patternX = particleIndex.sub(cellIndexInBlob.mul(particlesPerCell)).sub(patternY.mul(2));
  const cellY = cellIndexInBlob.div(blobW).floor();
  const cellX = cellIndexInBlob.sub(cellY.mul(blobW));
  const offset = vec2(patternX.mul(0.5).add(0.25), patternY.mul(0.5).add(0.25));
  position.assign(vec2(cellX.add(blobX), cellY.add(blobY)).add(offset));
  velocity.assign(vec2(0));
});

const clearUTransfer = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, uW, uH);
  const i = uIndex(p);
  atomicStore(uTemp.element(i), int(0));
  atomicStore(uWeightTemp.element(i), int(0));
})();

const clearVTransfer = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, vW, vH);
  const i = vIndex(p);
  atomicStore(vTemp.element(i), int(0));
  atomicStore(vWeightTemp.element(i), int(0));
})();

const clearCellTransfer = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, simW, simH);
  atomicStore(cellTemp.element(cellIndex(p)), int(0));
})();

const particleToGrid = Fn(() => {
  const position = particlePositions.element(instanceIndex);
  const velocity = particleVelocities.element(instanceIndex);
  const pc = position.sub(0.5);
  const c00 = clampCellCoord(pc.floor());
  const c10 = clampCellCoord(c00.add(vec2(1, 0)));
  const c01 = clampCellCoord(c00.add(vec2(0, 1)));
  const c11 = clampCellCoord(c00.add(vec2(1, 1)));
  const fc = pc.sub(c00);
  const cw00 = fc.x.oneMinus().mul(fc.y.oneMinus());
  const cw10 = fc.x.mul(fc.y.oneMinus());
  const cw01 = fc.x.oneMinus().mul(fc.y);
  const cw11 = fc.x.mul(fc.y);
  atomicAdd(cellTemp.element(cellIndex(c00)), int(cw00.mul(atomicScale)));
  atomicAdd(cellTemp.element(cellIndex(c10)), int(cw10.mul(atomicScale)));
  atomicAdd(cellTemp.element(cellIndex(c01)), int(cw01.mul(atomicScale)));
  atomicAdd(cellTemp.element(cellIndex(c11)), int(cw11.mul(atomicScale)));

  const pu = vec2(position.x, position.y.sub(0.5));
  const u00 = clampUCoord(pu.floor());
  const u10 = clampUCoord(u00.add(vec2(1, 0)));
  const u01 = clampUCoord(u00.add(vec2(0, 1)));
  const u11 = clampUCoord(u00.add(vec2(1, 1)));
  const fu = pu.sub(u00);
  const uw00 = fu.x.oneMinus().mul(fu.y.oneMinus());
  const uw10 = fu.x.mul(fu.y.oneMinus());
  const uw01 = fu.x.oneMinus().mul(fu.y);
  const uw11 = fu.x.mul(fu.y);

  atomicAdd(uTemp.element(uIndex(u00)), int(velocity.x.mul(uw00).mul(atomicScale)));
  atomicAdd(uWeightTemp.element(uIndex(u00)), int(uw00.mul(atomicScale)));
  atomicAdd(uTemp.element(uIndex(u10)), int(velocity.x.mul(uw10).mul(atomicScale)));
  atomicAdd(uWeightTemp.element(uIndex(u10)), int(uw10.mul(atomicScale)));
  atomicAdd(uTemp.element(uIndex(u01)), int(velocity.x.mul(uw01).mul(atomicScale)));
  atomicAdd(uWeightTemp.element(uIndex(u01)), int(uw01.mul(atomicScale)));
  atomicAdd(uTemp.element(uIndex(u11)), int(velocity.x.mul(uw11).mul(atomicScale)));
  atomicAdd(uWeightTemp.element(uIndex(u11)), int(uw11.mul(atomicScale)));

  const pv = vec2(position.x.sub(0.5), position.y);
  const v00 = clampVCoord(pv.floor());
  const v10 = clampVCoord(v00.add(vec2(1, 0)));
  const v01 = clampVCoord(v00.add(vec2(0, 1)));
  const v11 = clampVCoord(v00.add(vec2(1, 1)));
  const fv = pv.sub(v00);
  const vw00 = fv.x.oneMinus().mul(fv.y.oneMinus());
  const vw10 = fv.x.mul(fv.y.oneMinus());
  const vw01 = fv.x.oneMinus().mul(fv.y);
  const vw11 = fv.x.mul(fv.y);

  atomicAdd(vTemp.element(vIndex(v00)), int(velocity.y.mul(vw00).mul(atomicScale)));
  atomicAdd(vWeightTemp.element(vIndex(v00)), int(vw00.mul(atomicScale)));
  atomicAdd(vTemp.element(vIndex(v10)), int(velocity.y.mul(vw10).mul(atomicScale)));
  atomicAdd(vWeightTemp.element(vIndex(v10)), int(vw10.mul(atomicScale)));
  atomicAdd(vTemp.element(vIndex(v01)), int(velocity.y.mul(vw01).mul(atomicScale)));
  atomicAdd(vWeightTemp.element(vIndex(v01)), int(vw01.mul(atomicScale)));
  atomicAdd(vTemp.element(vIndex(v11)), int(velocity.y.mul(vw11).mul(atomicScale)));
  atomicAdd(vWeightTemp.element(vIndex(v11)), int(vw11.mul(atomicScale)));
});

const normalizeUGrid = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, uW, uH);
  const w = float(atomicLoad(uWeightTemp.element(uIndex(p))));
  const valid = w.min(1);
  const value = float(atomicLoad(uTemp.element(uIndex(p)))).div(w.max(1)).mul(valid);
  textureStore(uB, p, vec4(value, 0, 0, 0));
})();

const normalizeVGrid = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, vW, vH);
  const w = float(atomicLoad(vWeightTemp.element(vIndex(p))));
  const valid = w.min(1);
  const value = float(atomicLoad(vTemp.element(vIndex(p)))).div(w.max(1)).mul(valid);
  textureStore(vB, p, vec4(value, 0, 0, 0));
})();

const buildLiquidMask = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, simW, simH);
  const density = float(atomicLoad(cellTemp.element(cellIndex(p)))).div(atomicScale);
  const liquid = density.greaterThan(liquidThreshold).select(1, 0);
  textureStore(liquidMask, p, vec4(liquid, 0, 0, 0));
})();

const storeDivergence = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, simW, simH);
  const liquid = textureLoad(liquidMask, p).x;
  const uLeft = textureLoad(uA, vec2(p.x, p.y)).x;
  const uRight = textureLoad(uA, vec2(p.x.add(1), p.y)).x;
  const vBottom = textureLoad(vA, vec2(p.x, p.y)).x;
  const vTop = textureLoad(vA, vec2(p.x, p.y.add(1))).x;
  const div = uRight.sub(uLeft).add(vTop.sub(vBottom));
  textureStore(divergence, p, vec4(div.mul(liquid), 0, 0, 0));
})();

const jacobiPressure = ({ pressureSrc, pressureDst }) =>
  Fn(() => {
    const p = vec2(globalId.xy);
    guardBounds(p, simW, simH);
    const liquid = textureLoad(liquidMask, p).x;
    const leftAvail = p.x.min(1);
    const rightAvail = float(simW - 1).sub(p.x).min(1);
    const bottomAvail = p.y.min(1);
    const topAvail = float(simH - 1).sub(p.y).min(1);
    const pLeftCoord = clampCellCoord(vec2(p.x.sub(1), p.y));
    const pRightCoord = clampCellCoord(vec2(p.x.add(1), p.y));
    const pBottomCoord = clampCellCoord(vec2(p.x, p.y.sub(1)));
    const pTopCoord = clampCellCoord(vec2(p.x, p.y.add(1)));
    const pLeft = textureLoad(pressureSrc, pLeftCoord).x.mul(textureLoad(liquidMask, pLeftCoord).x).mul(leftAvail);
    const pRight = textureLoad(pressureSrc, pRightCoord).x.mul(textureLoad(liquidMask, pRightCoord).x).mul(rightAvail);
    const pBottom = textureLoad(pressureSrc, pBottomCoord).x.mul(textureLoad(liquidMask, pBottomCoord).x).mul(bottomAvail);
    const pTop = textureLoad(pressureSrc, pTopCoord).x.mul(textureLoad(liquidMask, pTopCoord).x).mul(topAvail);
    const denom = leftAvail.add(rightAvail).add(bottomAvail).add(topAvail).max(1);
    const pCenter = textureLoad(pressureSrc, p).x;
    const pJacobi = pLeft.add(pRight).add(pBottom).add(pTop).sub(textureLoad(divergence, p).x).div(denom);
    const pNext = pCenter.mul(float(1).sub(uniforms.underrelaxation)).add(pJacobi.mul(uniforms.underrelaxation)).mul(liquid);
    textureStore(pressureDst, p, vec4(pNext, 0, 0, 0));
  })();

const applyForcesU = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, uW, uH);
  const faceMask = uFaceMask(p);
  const facePos = vec2(p.x, p.y.add(0.5));
  const mousePos = vec2(
    uniforms.mouse.x.mul(uniforms.mouseToSim.x),
    uniforms.mouse.y.mul(uniforms.mouseToSim.y)
  );
  const mouseDelta = uniforms.mouse.xy.sub(uniforms.mouse.zw).mul(uniforms.mouseToSim);
  const mouseVel = mouseDelta.mul(2).div(uniforms.sdt);
  const mouseDist = facePos.distance(mousePos);
  const simRadius = uniforms.interactionRadius.mul(uniforms.mouseToSim.x.min(uniforms.mouseToSim.y));
  const falloff = mouseDist.div(simRadius).oneMinus().max(0).pow(2);
  const mouseForce = uniforms.mouseDown.select(mouseVel.x.mul(falloff), 0).mul(uniforms.sdt).mul(faceMask);
  const damping = uniforms.sdt.mul(dampingRate).oneMinus().max(0);
  const u = textureLoad(uB, p).x;
  const uNext = u.mul(damping).add(mouseForce).mul(faceMask);
  textureStore(uA, p, vec4(uNext, 0, 0, 0));
})();

const applyForcesV = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, vW, vH);
  const faceMask = vFaceMask(p);
  const facePos = vec2(p.x.add(0.5), p.y);
  const mousePos = vec2(
    uniforms.mouse.x.mul(uniforms.mouseToSim.x),
    uniforms.mouse.y.mul(uniforms.mouseToSim.y)
  );
  const mouseDelta = uniforms.mouse.xy.sub(uniforms.mouse.zw).mul(uniforms.mouseToSim);
  const mouseVel = mouseDelta.mul(2).div(uniforms.sdt);
  const mouseDist = facePos.distance(mousePos);
  const simRadius = uniforms.interactionRadius.mul(uniforms.mouseToSim.x.min(uniforms.mouseToSim.y));
  const falloff = mouseDist.div(simRadius).oneMinus().max(0).pow(2);
  const mouseForce = uniforms.mouseDown.select(mouseVel.y.mul(falloff), 0).mul(uniforms.sdt).mul(faceMask);
  const gravityForce = gravity.y.mul(uniforms.sdt).mul(faceMask);
  const damping = uniforms.sdt.mul(dampingRate).oneMinus().max(0);
  const v = textureLoad(vB, p).x;
  const vNext = v.mul(damping).add(mouseForce).add(gravityForce).mul(faceMask);
  textureStore(vA, p, vec4(vNext, 0, 0, 0));
})();

const projectU = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, uW, uH);
  const leftCell = clampCellCoord(vec2(p.x.sub(1), p.y));
  const rightCell = clampCellCoord(vec2(p.x, p.y));
  const faceMask = uFaceMask(p);
  const leftMask = textureLoad(liquidMask, leftCell).x;
  const rightMask = textureLoad(liquidMask, rightCell).x;
  const u = textureLoad(uA, p).x;
  const pLeft = textureLoad(pressureA, leftCell).x.mul(leftMask);
  const pRight = textureLoad(pressureA, rightCell).x.mul(rightMask);
  const uNext = u.sub(pRight.sub(pLeft)).mul(faceMask);
  textureStore(uC, p, vec4(uNext, 0, 0, 0));
})();

const projectV = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, vW, vH);
  const bottomCell = clampCellCoord(vec2(p.x, p.y.sub(1)));
  const topCell = clampCellCoord(vec2(p.x, p.y));
  const faceMask = vFaceMask(p);
  const bottomMask = textureLoad(liquidMask, bottomCell).x;
  const topMask = textureLoad(liquidMask, topCell).x;
  const v = textureLoad(vA, p).x;
  const pBottom = textureLoad(pressureA, bottomCell).x.mul(bottomMask);
  const pTop = textureLoad(pressureA, topCell).x.mul(topMask);
  const vNext = v.sub(pTop.sub(pBottom)).mul(faceMask);
  textureStore(vC, p, vec4(vNext, 0, 0, 0));
})();

const buildUValidity = ({ dst }) =>
  Fn(() => {
    const p = vec2(globalId.xy);
    guardBounds(p, uW, uH);
    textureStore(dst, p, vec4(uFaceMask(p), 0, 0, 0));
  })();

const buildVValidity = ({ dst }) =>
  Fn(() => {
    const p = vec2(globalId.xy);
    guardBounds(p, vW, vH);
    textureStore(dst, p, vec4(vFaceMask(p), 0, 0, 0));
  })();

const extrapolateU = ({ src, srcValid, dst, dstValid }) =>
  Fn(() => {
    const p = vec2(globalId.xy);
    guardBounds(p, uW, uH);
    const valid = textureLoad(srcValid, p).x;
    const pLeft = clampUCoord(vec2(p.x.sub(1), p.y));
    const pRight = clampUCoord(vec2(p.x.add(1), p.y));
    const pBottom = clampUCoord(vec2(p.x, p.y.sub(1)));
    const pTop = clampUCoord(vec2(p.x, p.y.add(1)));
    const vLeft = textureLoad(srcValid, pLeft).x;
    const vRight = textureLoad(srcValid, pRight).x;
    const vBottom = textureLoad(srcValid, pBottom).x;
    const vTop = textureLoad(srcValid, pTop).x;
    const count = vLeft.add(vRight).add(vBottom).add(vTop).max(1);
    const avg = textureLoad(src, pLeft).x.mul(vLeft)
      .add(textureLoad(src, pRight).x.mul(vRight))
      .add(textureLoad(src, pBottom).x.mul(vBottom))
      .add(textureLoad(src, pTop).x.mul(vTop))
      .div(count);
    const next = textureLoad(src, p).x.mul(valid).add(avg.mul(valid.oneMinus()));
    const propagatedValid = valid.max(vLeft.max(vRight).max(vBottom).max(vTop));
    textureStore(dst, p, vec4(next, 0, 0, 0));
    textureStore(dstValid, p, vec4(propagatedValid, 0, 0, 0));
  })();

const extrapolateV = ({ src, srcValid, dst, dstValid }) =>
  Fn(() => {
    const p = vec2(globalId.xy);
    guardBounds(p, vW, vH);
    const valid = textureLoad(srcValid, p).x;
    const pLeft = clampVCoord(vec2(p.x.sub(1), p.y));
    const pRight = clampVCoord(vec2(p.x.add(1), p.y));
    const pBottom = clampVCoord(vec2(p.x, p.y.sub(1)));
    const pTop = clampVCoord(vec2(p.x, p.y.add(1)));
    const vLeft = textureLoad(srcValid, pLeft).x;
    const vRight = textureLoad(srcValid, pRight).x;
    const vBottom = textureLoad(srcValid, pBottom).x;
    const vTop = textureLoad(srcValid, pTop).x;
    const count = vLeft.add(vRight).add(vBottom).add(vTop).max(1);
    const avg = textureLoad(src, pLeft).x.mul(vLeft)
      .add(textureLoad(src, pRight).x.mul(vRight))
      .add(textureLoad(src, pBottom).x.mul(vBottom))
      .add(textureLoad(src, pTop).x.mul(vTop))
    .div(count);
    const next = textureLoad(src, p).x.mul(valid).add(avg.mul(valid.oneMinus()));
    const propagatedValid = valid.max(vLeft.max(vRight).max(vBottom).max(vTop));
    textureStore(dst, p, vec4(next, 0, 0, 0));
    textureStore(dstValid, p, vec4(propagatedValid, 0, 0, 0));
  })();

const storeProjectedDivergence = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, simW, simH);
  const liquid = textureLoad(liquidMask, p).x;
  const uLeft = textureLoad(uC, vec2(p.x, p.y)).x;
  const uRight = textureLoad(uC, vec2(p.x.add(1), p.y)).x;
  const vBottom = textureLoad(vC, vec2(p.x, p.y)).x;
  const vTop = textureLoad(vC, vec2(p.x, p.y.add(1))).x;
  const div = uRight.sub(uLeft).add(vTop.sub(vBottom));
  textureStore(projectedDivergence, p, vec4(div.mul(liquid), 0, 0, 0));
})();

const storeDebugDisplay = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, simW, simH);
  const before = textureLoad(divergence, p).x.abs().mul(8).clamp(0, 1);
  const after = textureLoad(projectedDivergence, p).x.abs().mul(8).clamp(0, 1);
  textureStore(displayVelocity, p, vec4(before, after, 0, 0));
})();

const reconstructDisplayVelocity = Fn(() => {
  const p = vec2(globalId.xy);
  guardBounds(p, simW, simH);
  const uLeft = textureLoad(uC, vec2(p.x, p.y)).x;
  const uRight = textureLoad(uC, vec2(p.x.add(1), p.y)).x;
  const vBottom = textureLoad(vC, vec2(p.x, p.y)).x;
  const vTop = textureLoad(vC, vec2(p.x, p.y.add(1))).x;
  const velocity = vec2(uLeft.add(uRight), vBottom.add(vTop)).mul(0.5);
  textureStore(displayVelocity, p, vec4(velocity, 0, 0));
})();

const gridToParticleAndAdvect = Fn(() => {
  const position = particlePositions.element(instanceIndex);
  const particleVelocity = particleVelocities.element(instanceIndex);
  const PIC = sampleVelocity(uC, vC, position);
  const FLIP = particleVelocity.add(PIC).sub(sampleVelocity(uB, vB, position));
  particleVelocity.assign(mix(PIC, FLIP, uniforms.flipAmount));
  position.assign(clampParticlePos(position.add(particleVelocity.mul(uniforms.sdt))));
});

const zeroCellScalar = ({ dst }) =>
  Fn(() => {
    const p = vec2(globalId.xy);
    guardBounds(p, simW, simH);
    textureStore(dst, p, vec4(0, 0, 0, 0));
  })();

const zeroUField = ({ dst }) =>
  Fn(() => {
    const p = vec2(globalId.xy);
    guardBounds(p, uW, uH);
    textureStore(dst, p, vec4(0, 0, 0, 0));
  })();

const zeroVField = ({ dst }) =>
  Fn(() => {
    const p = vec2(globalId.xy);
    guardBounds(p, vW, vH);
    textureStore(dst, p, vec4(0, 0, 0, 0));
  })();

const clearPressureACompute = zeroCellScalar({ dst: pressureA }).computeKernel(WG);
const clearPressureBCompute = zeroCellScalar({ dst: pressureB }).computeKernel(WG);
const clearDivergenceCompute = zeroCellScalar({ dst: divergence }).computeKernel(WG);
const clearProjectedDivergenceCompute = zeroCellScalar({ dst: projectedDivergence }).computeKernel(WG);
const clearLiquidMaskCompute = zeroCellScalar({ dst: liquidMask }).computeKernel(WG);
const clearDisplayVelocityCompute = zeroCellScalar({ dst: displayVelocity }).computeKernel(WG);
const clearUACompute = zeroUField({ dst: uA }).computeKernel(WG);
const clearUBCompute = zeroUField({ dst: uB }).computeKernel(WG);
const clearUCCompute = zeroUField({ dst: uC }).computeKernel(WG);
const clearUValidACompute = zeroUField({ dst: uValidA }).computeKernel(WG);
const clearUValidBCompute = zeroUField({ dst: uValidB }).computeKernel(WG);
const clearVACompute = zeroVField({ dst: vA }).computeKernel(WG);
const clearVBCompute = zeroVField({ dst: vB }).computeKernel(WG);
const clearVCCompute = zeroVField({ dst: vC }).computeKernel(WG);
const clearVValidACompute = zeroVField({ dst: vValidA }).computeKernel(WG);
const clearVValidBCompute = zeroVField({ dst: vValidB }).computeKernel(WG);

const clearUTransferCompute = clearUTransfer.computeKernel(WG);
const clearVTransferCompute = clearVTransfer.computeKernel(WG);
const clearCellTransferCompute = clearCellTransfer.computeKernel(WG);
const scatterParticlesCompute = scatterParticles().compute(particleAmount, WG);
const particleToGridCompute = particleToGrid().compute(particleAmount, WG);
const normalizeUCompute = normalizeUGrid.computeKernel(WG);
const normalizeVCompute = normalizeVGrid.computeKernel(WG);
const buildLiquidMaskCompute = buildLiquidMask.computeKernel(WG);
const divergenceCompute = storeDivergence.computeKernel(WG);
const projectedDivergenceCompute = storeProjectedDivergence.computeKernel(WG);
const debugDisplayCompute = storeDebugDisplay.computeKernel(WG);
const pressureABCompute = jacobiPressure({ pressureSrc: pressureA, pressureDst: pressureB }).computeKernel(WG);
const pressureBACompute = jacobiPressure({ pressureSrc: pressureB, pressureDst: pressureA }).computeKernel(WG);
const applyForcesUCompute = applyForcesU.computeKernel(WG);
const applyForcesVCompute = applyForcesV.computeKernel(WG);
const projectUCompute = projectU.computeKernel(WG);
const projectVCompute = projectV.computeKernel(WG);
const buildUValidityCompute = buildUValidity({ dst: uValidA }).computeKernel(WG);
const buildVValidityCompute = buildVValidity({ dst: vValidA }).computeKernel(WG);
const extrapolateUCACompute = extrapolateU({ src: uC, srcValid: uValidA, dst: uA, dstValid: uValidB }).computeKernel(WG);
const extrapolateVCACompute = extrapolateV({ src: vC, srcValid: vValidA, dst: vA, dstValid: vValidB }).computeKernel(WG);
const extrapolateUACCompute = extrapolateU({ src: uA, srcValid: uValidB, dst: uC, dstValid: uValidA }).computeKernel(WG);
const extrapolateVACCompute = extrapolateV({ src: vA, srcValid: vValidB, dst: vC, dstValid: vValidA }).computeKernel(WG);
const extrapolationSequence = [
  [extrapolateUCACompute, extrapolateVCACompute],
  [extrapolateUACCompute, extrapolateVACCompute],
  [extrapolateUCACompute, extrapolateVCACompute],
  [extrapolateUACCompute, extrapolateVACCompute],
];
const reconstructDisplayVelocityCompute = reconstructDisplayVelocity.computeKernel(WG);
const gridToParticleCompute = gridToParticleAndAdvect().compute(particleAmount, WG);

const gridMaterial = new THREE.MeshBasicMaterial({ transparent: true });
gridMaterial.colorNode = Fn(() => {
  const v = texture(displayVelocity).xy;
  if (debugPressure) {
    return vec4(v.x, v.y, 0, 1);
  }
  const speed = v.length();
  const maxSpeed = float(0.5).div(float(1 / 60).div(substeps));
  const t = speed.div(maxSpeed).clamp(0, 1);
  const r = t.mul(0.5).oneMinus().mul(t);
  const g = t.oneMinus().mul(t);
  const b = t.mul(t);
  return vec4(b, g, r, 1);
})();

const gridQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), gridMaterial);
gridQuad.renderOrder = 0;
scene.add(gridQuad);

const particleMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95 });
particleMaterial.positionNode = Fn(() => {
  const p = particlePositions.element(instanceIndex);
  const particleUV = p.div(simSize);
  const center = vec3(particleUV.x.mul(2).sub(1), particleUV.y.mul(-2).add(1), 0);
  return center.add(positionGeometry);
})();

const particleMesh = new THREE.InstancedMesh(new THREE.CircleGeometry(0.006, 8), particleMaterial, particleAmount);
particleMesh.frustumCulled = false;
particleMesh.renderOrder = 1;
const identity = new THREE.Matrix4();
for (let i = 0; i < particleAmount; i++) {
  particleMesh.setMatrixAt(i, identity);
}
scene.add(particleMesh);

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
  if (canvas.hasPointerCapture(e.pointerId)) {
    canvas.releasePointerCapture(e.pointerId);
  }
  pointer.down = false;
});

canvas.addEventListener('pointercancel', () => {
  pointer.down = false;
});

canvas.addEventListener('lostpointercapture', () => {
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
  renderer.compute(clearPressureACompute, cellDispatch);
  renderer.compute(clearPressureBCompute, cellDispatch);
  renderer.compute(clearDivergenceCompute, cellDispatch);
  renderer.compute(clearProjectedDivergenceCompute, cellDispatch);
  renderer.compute(clearLiquidMaskCompute, cellDispatch);
  renderer.compute(clearDisplayVelocityCompute, cellDispatch);
  renderer.compute(clearUACompute, uDispatch);
  renderer.compute(clearUBCompute, uDispatch);
  renderer.compute(clearUCCompute, uDispatch);
  renderer.compute(clearUValidACompute, uDispatch);
  renderer.compute(clearUValidBCompute, uDispatch);
  renderer.compute(clearVACompute, vDispatch);
  renderer.compute(clearVBCompute, vDispatch);
  renderer.compute(clearVCCompute, vDispatch);
  renderer.compute(clearVValidACompute, vDispatch);
  renderer.compute(clearVValidBCompute, vDispatch);
  renderer.compute(scatterParticlesCompute);

  renderer.setAnimationLoop((timeMs) => {
    const dt = Math.min(Math.max((timeMs - lastTime) * 0.001, 1 / 240), 1 / 30);
    lastTime = timeMs;

    uniforms.sdt.value = dt / substeps;
    uniforms.mouseToSim.value.set(simW / canvas.clientWidth, simH / canvas.clientHeight);
    uniforms.mouse.value.set(pointer.x, pointer.y, pointer.px, pointer.py);
    uniforms.mouseDown.value = pointer.down;

    for (let i = 0; i < substeps; i++) {
      renderer.compute(clearUTransferCompute, uDispatch);
      renderer.compute(clearVTransferCompute, vDispatch);
      renderer.compute(clearCellTransferCompute, cellDispatch);
      renderer.compute(particleToGridCompute);
      renderer.compute(normalizeUCompute, uDispatch);
      renderer.compute(normalizeVCompute, vDispatch);
      renderer.compute(buildLiquidMaskCompute, cellDispatch);
      renderer.compute(applyForcesUCompute, uDispatch);
      renderer.compute(applyForcesVCompute, vDispatch);
      renderer.compute(divergenceCompute, cellDispatch);
      renderer.compute(clearPressureACompute, cellDispatch);
      renderer.compute(clearPressureBCompute, cellDispatch);
      for (let j = 0; j < params.pressureLoops; j++) {
        renderer.compute(pressureABCompute, cellDispatch);
        renderer.compute(pressureBACompute, cellDispatch);
      }
      renderer.compute(projectUCompute, uDispatch);
      renderer.compute(projectVCompute, vDispatch);
      renderer.compute(buildUValidityCompute, uDispatch);
      renderer.compute(buildVValidityCompute, vDispatch);
      for (const [uExtrapolate, vExtrapolate] of extrapolationSequence) {
        renderer.compute(uExtrapolate, uDispatch);
        renderer.compute(vExtrapolate, vDispatch);
      }
      if (debugPressure) {
        renderer.compute(projectedDivergenceCompute, cellDispatch);
        renderer.compute(debugDisplayCompute, cellDispatch);
      } else {
        renderer.compute(reconstructDisplayVelocityCompute, cellDispatch);
      }
      renderer.compute(gridToParticleCompute);
    }

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
