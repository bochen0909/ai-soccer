import { PHYSICS_TICK_SECONDS } from "../engine/physics.js";

const STEP_MS = PHYSICS_TICK_SECONDS * 1_000;
// Display timestamps and the tick interval can round in opposite directions.
const TIMESTAMP_TOLERANCE_MS = 1e-7;

export interface FixedStepOptions {
  maxStepsPerFrame?: number;
  maxGapMs?: number;
}

export interface FixedStepResult<Snapshot> {
  snapshot: Snapshot;
  align?: boolean;
  finished?: boolean;
}

export interface FixedStepFrame<Snapshot> {
  previous: Snapshot;
  current: Snapshot;
  alpha: number;
  steps: number;
  pauseReason: "gap" | "backlog" | null;
  finished: boolean;
}

export class FixedStepClock<Snapshot> {
  private previous: Snapshot;
  private current: Snapshot;
  private lastTimestamp: number | undefined;
  private accumulatorMs = 0;
  private finished = false;
  private readonly maxStepsPerFrame: number;
  private readonly maxGapMs: number;

  constructor(initial: Snapshot, options: FixedStepOptions = {}) {
    this.previous = initial;
    this.current = initial;
    this.maxStepsPerFrame = options.maxStepsPerFrame ?? 8;
    this.maxGapMs = options.maxGapMs ?? 250;
    if (!Number.isSafeInteger(this.maxStepsPerFrame) || this.maxStepsPerFrame < 1) {
      throw new RangeError("The frame step limit must be a positive integer.");
    }
    if (!Number.isFinite(this.maxGapMs) || this.maxGapMs <= 0) {
      throw new RangeError("The frame gap limit must be positive and finite.");
    }
  }

  /** Rebase wall time and interpolation on the current or newly reset match. */
  align(snapshot?: Snapshot): void {
    if (snapshot !== undefined) {
      this.current = snapshot;
      this.finished = false;
    }
    this.previous = this.current;
    this.lastTimestamp = undefined;
    this.accumulatorMs = 0;
  }

  frame(): FixedStepFrame<Snapshot> {
    return {
      previous: this.previous,
      current: this.current,
      alpha: this.accumulatorMs / STEP_MS,
      steps: 0,
      pauseReason: null,
      finished: this.finished,
    };
  }

  advance(nowMs: number, step: () => FixedStepResult<Snapshot>): FixedStepFrame<Snapshot> {
    if (this.finished) return this.frame();

    const elapsedMs = nowMs - (this.lastTimestamp ?? nowMs);
    this.lastTimestamp = nowMs;
    if (elapsedMs > this.maxGapMs) {
      this.align();
      return { ...this.frame(), pauseReason: "gap" };
    }

    this.accumulatorMs += Math.max(0, elapsedMs);
    let steps = 0;
    while (this.hasTick() && steps < this.maxStepsPerFrame) {
      const result = step();
      this.previous = result.align || result.finished ? result.snapshot : this.current;
      this.current = result.snapshot;
      this.accumulatorMs = Math.max(0, this.accumulatorMs - STEP_MS);
      steps += 1;
      if (result.finished) {
        this.finished = true;
        this.align();
        return { ...this.frame(), steps };
      }
    }

    if (this.hasTick()) {
      this.align();
      return { ...this.frame(), steps, pauseReason: "backlog" };
    }

    return { ...this.frame(), steps };
  }

  private hasTick(): boolean {
    return this.accumulatorMs + TIMESTAMP_TOLERANCE_MS >= STEP_MS;
  }
}
