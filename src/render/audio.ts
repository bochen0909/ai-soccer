import type { MatchEvent } from "../contracts/index.js";

export class MatchAudio {
  private context?: AudioContext;
  private enabled = true;

  unlock(): void {
    if (!this.enabled || !("AudioContext" in window)) return;
    try {
      this.context ??= new AudioContext();
      if (this.context.state === "suspended") void this.context.resume().catch(() => {});
    } catch {
      // Playback depends on browser support and user gesture permissions.
    }
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (enabled) this.unlock();
  }

  play(events: readonly MatchEvent[]): void {
    const context = this.context;
    if (!this.enabled || !context || context.state !== "running") return;
    for (const event of events) {
      if (event.kind === "goal") {
        this.tone(392, 392, 0.13, 0.09, 0);
        this.tone(494, 494, 0.13, 0.09, 0.11);
        this.tone(587, 587, 0.22, 0.11, 0.22);
      } else if (event.kind === "actionContact" && event.outcome !== "missed") {
        this.tone(140, 53, 0.08, 0.11, 0);
      } else if (event.kind === "ballExit" || (event.kind === "phaseChanged" &&
        (event.phase.kind === "halftime" || event.phase.kind === "fullTime"))) {
        this.tone(1460, 1780, 0.15, 0.04, 0);
      }
    }
  }

  private tone(from: number, to: number, duration: number, volume: number, delay: number): void {
    const context = this.context!;
    const time = context.currentTime + delay;
    const note = context.createOscillator();
    const envelope = context.createGain();
    note.type = "sine";
    note.frequency.setValueAtTime(from, time);
    note.frequency.exponentialRampToValueAtTime(to, time + duration);
    envelope.gain.setValueAtTime(0.001, time);
    envelope.gain.exponentialRampToValueAtTime(volume, time + 0.01);
    envelope.gain.exponentialRampToValueAtTime(0.001, time + duration);
    note.connect(envelope).connect(context.destination);
    note.start(time);
    note.stop(time + duration);
  }
}
