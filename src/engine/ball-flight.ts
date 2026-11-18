import type { BallConfig } from "../contracts/index.js";

/**
 * Ball motion shared by the engine and by everyone who predicts it (CPU, learner decoder, keyboard aim assist).
 * Rolling slows by a constant grass resistance plus air drag proportional to speed squared; flight feels drag and
 * gravity. The closed forms solve v dv/dx = -(a + k v^2), with a the rolling deceleration and k the drag factor.
 */

type BallMotion = Pick<BallConfig, "groundDecelerationMps2" | "airDragPerM">;

const EPSILON = 1e-9;

/** Speed of a rolling ball after it has covered `distance`, or 0 if it stops first. */
export function rollSpeedAfter(speed: number, distance: number, ball: BallMotion): number {
  const { groundDecelerationMps2: a, airDragPerM: k } = ball;
  if (k <= EPSILON) return Math.sqrt(Math.max(0, speed * speed - 2 * a * distance));
  const squared = (speed * speed + a / k) * Math.exp(-2 * k * distance) - a / k;
  return Math.sqrt(Math.max(0, squared));
}

/** Launch speed a rolling ball needs to still be moving at `arrival` after `distance`. */
export function rollLaunchSpeed(distance: number, arrival: number, ball: BallMotion): number {
  const { groundDecelerationMps2: a, airDragPerM: k } = ball;
  if (k <= EPSILON) return Math.sqrt(arrival * arrival + 2 * a * distance);
  return Math.sqrt((arrival * arrival + a / k) * Math.exp(2 * k * distance) - a / k);
}

/** Distance a rolling ball launched at `speed` covers in `seconds`, stopping where friction and drag halt it. */
export function rollDistance(speed: number, seconds: number, ball: BallMotion): number {
  const { groundDecelerationMps2: a, airDragPerM: k } = ball;
  if (speed <= EPSILON || seconds <= 0) return 0;
  if (a <= EPSILON) return k <= EPSILON ? speed * seconds : Math.log(1 + k * speed * seconds) / k;
  if (k <= EPSILON) {
    const time = Math.min(seconds, speed / a);
    return speed * time - 0.5 * a * time * time;
  }
  // v(t) = sqrt(a/k) tan(theta0 - w t) with w = sqrt(a k); x(t) = ln(cos(theta0 - w t) / cos(theta0)) / k.
  const w = Math.sqrt(a * k);
  const theta0 = Math.atan(speed * Math.sqrt(k / a));
  const theta = Math.max(0, theta0 - w * seconds);
  return Math.log(Math.cos(theta) / Math.cos(theta0)) / k;
}

/** Horizontal distance a ball in flight covers in `seconds` under drag alone (gravity is left to the caller). */
export function flightDistance(speed: number, seconds: number, ball: BallMotion): number {
  const k = ball.airDragPerM;
  if (seconds <= 0) return 0;
  return k <= EPSILON ? speed * seconds : Math.log(1 + k * speed * seconds) / k;
}

/**
 * Height of a kicked ball as it passes `range` metres downfield, or -Infinity if it lands first. Integrates drag and
 * gravity in small steps; used to pick lofted-pass pace.
 */
function heightAtRange(speed: number, elevation: number, range: number, ball: BallConfig, startHeight: number): number {
  const step = 1 / 120;
  let x = 0, z = startHeight, vx = speed * Math.cos(elevation), vz = speed * Math.sin(elevation);
  for (let t = 0; t < 8; t += step) {
    const v = Math.hypot(vx, vz);
    const drag = 1 / (1 + ball.airDragPerM * v * step);
    vx *= drag;
    vz = vz * drag - ball.gravityMps2 * step;
    const nx = x + vx * step, nz = z + vz * step;
    if (nx >= range) return z + (nz - z) * ((range - x) / Math.max(EPSILON, nx - x));
    if (nz < ball.radiusM && vz < 0) return -Infinity;
    x = nx; z = nz;
  }
  return -Infinity;
}

/** Launch speed for a kick at `elevation` that is at `height` when it reaches `range` (the ground for a landing). */
export function flightLaunchSpeed(
  range: number, elevation: number, ball: BallConfig, height = ball.radiusM, startHeight = ball.radiusM,
): number {
  let low = 0.5, high = ball.maxSpeedMps;
  if (heightAtRange(high, elevation, range, ball, startHeight) < height) return high;
  for (let i = 0; i < 30; i++) {
    const mid = (low + high) / 2;
    if (heightAtRange(mid, elevation, range, ball, startHeight) >= height) high = mid;
    else low = mid;
  }
  return high;
}

/**
 * Where a free ball will be after `seconds`: rolling with grass resistance and drag, or in flight with drag along its
 * path and gravity on its height (no bounce; the height stops at the ground).
 */
export function ballPositionAfter(
  position: { x: number; y: number; z: number }, velocity: { x: number; y: number; z: number }, seconds: number,
  ball: BallConfig,
): { x: number; y: number; z: number } {
  const ground = Math.hypot(velocity.x, velocity.y);
  const rolling = position.z <= ball.radiusM + 1e-6 && Math.abs(velocity.z) < 1e-6;
  const speed = rolling ? ground : Math.hypot(ground, velocity.z);
  const travelled = rolling ? rollDistance(ground, seconds, ball) : flightDistance(speed, seconds, ball);
  const scale = speed > EPSILON ? travelled / speed : 0;
  const z = rolling ? position.z
    : Math.max(ball.radiusM, position.z + velocity.z * scale - 0.5 * ball.gravityMps2 * seconds * seconds);
  return { x: position.x + velocity.x * scale, y: position.y + velocity.y * scale, z };
}
