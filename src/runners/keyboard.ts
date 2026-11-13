import type { MatchConfig, Vec2 } from "../contracts/index.js";
import type { ActionIntent, PassTarget } from "../controllers/contracts.js";
import type { HumanInput, SessionSnapshot } from "../session/contracts.js";

type Player = SessionSnapshot["players"][number];
type Key = "up" | "down" | "left" | "right" | "sprint" | "pass" | "shot" | "loft" | "defend" | "select";
type Action = { kind: "pass" | "loft" | "defend" } | { kind: "shot"; power: number };

const keys: Readonly<Record<string, Key>> = {
  w: "up", arrowup: "up", s: "down", arrowdown: "down",
  a: "left", arrowleft: "left", d: "right", arrowright: "right",
  shift: "sprint", j: "pass", k: "shot", l: "loft", " ": "defend", q: "select",
};
const neutral = { x: 0, y: 0 };
const chargeMilliseconds = 720;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function distance(first: Vec2, second: Vec2): number {
  return Math.hypot(second.x - first.x, second.y - first.y);
}

function direction(from: Vec2, to: Vec2, fallback: Vec2): Vec2 {
  const length = distance(from, to);
  return length > 1e-6 ? { x: (to.x - from.x) / length, y: (to.y - from.y) / length } : fallback;
}

function boundarySafeDirection(origin: Vec2, aim: Vec2, pitch: MatchConfig["pitch"]): Vec2 {
  const x = Math.abs(origin.x) >= pitch.lengthM / 2 - 1e-6 && origin.x * aim.x >= 0
    ? -Math.sign(origin.x) * 0.35 : aim.x;
  const y = Math.abs(origin.y) >= pitch.widthM / 2 - 1e-6 && origin.y * aim.y >= 0
    ? -Math.sign(origin.y) * 0.35 : aim.y;
  return { x, y };
}

function passTarget(
  snapshot: SessionSnapshot, player: Player, config: MatchConfig, aim: Vec2, maxDistance: number,
  groundKick = false,
): PassTarget | undefined {
  const origin = snapshot.ball.position;
  const needsClearLane = snapshot.phase.kind !== "live" &&
    !(snapshot.phase.kind === "kickoffSetup" && config.rules.directionalKickoff);
  let best: { pass: PassTarget; score: number } | undefined;
  for (const receiver of snapshot.players) {
    if (receiver.teamId !== player.teamId || receiver.playerId === player.playerId) continue;
    const target = {
      x: clamp(receiver.position.x + receiver.velocity.x * 0.25, -config.pitch.lengthM / 2, config.pitch.lengthM / 2),
      y: clamp(receiver.position.y + receiver.velocity.y * 0.25, -config.pitch.widthM / 2, config.pitch.widthM / 2),
    };
    const range = distance(origin, target);
    if (range < 1.2 || range > maxDistance) continue;
    const heading = direction(origin, target, aim);
    const fromPlayer = { x: origin.x - player.position.x, y: origin.y - player.position.y };
    if (groundKick && needsClearLane && fromPlayer.x * heading.x + fromPlayer.y * heading.y < -0.03) {
      continue;
    }
    const alignment = heading.x * aim.x + heading.y * aim.y;
    if (snapshot.phase.kind === "kickoffSetup" && config.rules.directionalKickoff && alignment <= 0) continue;
    let obstruction = 0;
    for (const opponent of snapshot.players) {
      if (opponent.teamId === player.teamId) continue;
      const dx = opponent.position.x - origin.x;
      const dy = opponent.position.y - origin.y;
      const along = dx * heading.x + dy * heading.y;
      if (along <= 0 || along >= range) continue;
      const clearance = Math.abs(dx * heading.y - dy * heading.x);
      obstruction = Math.max(obstruction, Math.max(0, 1.3 - clearance));
    }
    const score = alignment * 3.4 - range * 0.075 - obstruction * 0.9 -
      (receiver.definition.role === "goalkeeper" ? 0.35 : 0);
    if (!best || score > best.score) best = { pass: { receiverId: receiver.playerId, target }, score };
  }
  return best?.pass;
}

function groundPower(range: number, config: MatchConfig, speed: number, arrivalSpeed: number): number {
  return clamp(Math.sqrt(2 * config.ball.groundDecelerationMps2 * range + arrivalSpeed ** 2) / speed, 0, 0.96);
}

function kick(
  snapshot: SessionSnapshot, config: MatchConfig, player: Player, kind: "pass" | "loft", aim: Vec2,
): Pick<HumanInput, "request" | "pass"> {
  const { maxKickSpeedMps } = player.definition.abilities.physical;
  const maxSpeed = maxKickSpeedMps * (kind === "pass" ? 0.55 : 0.75);
  const rollout = maxSpeed ** 2 / (2 * Math.max(0.4, config.ball.groundDecelerationMps2));
  const maxDistance = kind === "pass" ? Math.min(25, rollout * 0.7) : Math.min(34, maxSpeed ** 2 / 12);
  const pass = passTarget(snapshot, player, config, aim, maxDistance, kind === "pass");
  const origin = snapshot.ball.position;
  const range = pass ? distance(origin, pass.target) : kind === "pass" ? 4 : 7;
  // Keep short passes controllable and give passes into empty space a short rollout.
  const arrivalSpeed = pass ? Math.min(3.6, 2 + range * 0.075) : 1.8;
  const heading = pass ? direction(origin, pass.target, aim) : aim;
  const fromPlayer = direction(player.position, origin, aim);
  const needsClearLane = snapshot.phase.kind !== "live" &&
    !(snapshot.phase.kind === "kickoffSetup" && config.rules.directionalKickoff);
  const safeHeading = kind === "pass" && needsClearLane &&
      fromPlayer.x * heading.x + fromPlayer.y * heading.y < 0 ? fromPlayer : heading;
  return {
    request: {
      kind: "kick", kickType: kind,
      direction: boundarySafeDirection(origin, safeHeading, config.pitch),
      power: kind === "pass" ? groundPower(range, config, maxSpeed, arrivalSpeed) : clamp(0.44 + range / 54, 0.44, 0.95),
      loft: kind === "pass" ? 0 : 0.36,
    },
    pass,
  };
}

function restart(
  snapshot: SessionSnapshot, config: MatchConfig, player: Player, action: Action, aim: Vec2,
): Pick<HumanInput, "request" | "pass"> {
  const { phase, tick } = snapshot;
  if ((phase.kind !== "kickoffSetup" && phase.kind !== "restartSetup") ||
      phase.takerId !== player.playerId || tick < phase.readyTick || tick >= phase.deadlineTick ||
      action.kind === "defend") return {};
  if (phase.kind === "restartSetup" && phase.restart === "throwIn") {
    const pass = passTarget(snapshot, player, config, aim, 15);
    const origin = snapshot.ball.position;
    const heading = pass ? direction(origin, pass.target, aim) : aim;
    return {
      request: {
        kind: "throwIn", direction: boundarySafeDirection(origin, heading, config.pitch),
        power: pass ? clamp(distance(origin, pass.target) / 15, 0.4, 0.85) : 0.53,
      },
    };
  }
  return kick(snapshot, config, player, phase.kind === "restartSetup" && action.kind === "loft" ? "loft" : "pass", aim);
}

function keeperRelease(
  snapshot: SessionSnapshot, config: MatchConfig, player: Player, action: Action, aim: Vec2,
): ActionIntent | undefined {
  if (action.kind === "defend") return undefined;
  const technique = action.kind === "loft" ? "throw" : "roll";
  const target = passTarget(snapshot, player, config, aim, technique === "roll" ? 11 : 20)?.target;
  const origin = player.position;
  const range = target ? distance(origin, target) : 8;
  const speed = player.definition.abilities.physical.maxKeeperRollSpeedMps;
  return {
    kind: "keeperRelease", technique, direction: target ? direction(origin, target, aim) : aim,
    power: technique === "roll" ? Math.max(0.3, groundPower(range, config, speed, Math.sqrt(18)))
      : clamp(range / 22, 0.42, 0.88),
  };
}

function selectPlayer(snapshot: SessionSnapshot, player: Player, cycle: boolean): Player {
  const { phase } = snapshot;
  const teammates = snapshot.players.filter((candidate) => candidate.teamId === player.teamId);
  if ((phase.kind === "kickoffSetup" || phase.kind === "restartSetup") && phase.teamId === player.teamId) {
    const taker = teammates.find((candidate) => candidate.playerId === phase.takerId);
    if (taker) return taker;
  }
  if (cycle) return teammates[(teammates.indexOf(player) + 1) % teammates.length];

  const { ball } = snapshot;
  if (ball.control.kind !== "free") {
    const { playerId } = ball.control;
    const holder = teammates.find((candidate) => candidate.playerId === playerId);
    if (holder) return holder;
  }
  const target = {
    x: ball.position.x + 0.25 * ball.velocity.x,
    y: ball.position.y + 0.25 * ball.velocity.y,
  };
  const timeToBall = (candidate: Player): number => {
    const speed = candidate.definition.abilities.physical.maxSpeedMps;
    return distance(candidate.position, target) / speed;
  };
  const outfield = teammates.filter((candidate) => candidate.definition.role === "outfield");
  return outfield.reduce((best, candidate) => timeToBall(candidate) < timeToBall(best) ? candidate : best,
    outfield.includes(player) ? player : outfield[0] ?? player);
}

function keyFor(event: KeyboardEvent): Key | undefined {
  const code = event.code?.toLowerCase() ?? "";
  const character = code.startsWith("key") ? code.slice(3) : event.key.toLowerCase();
  return Object.hasOwn(keys, character) ? keys[character] : undefined;
}

function tokenFor(event: KeyboardEvent): string {
  return event.code || event.key.toLowerCase();
}

export class KeyboardControls {
  private readonly held = new Map<string, Key>();
  private action: Action | undefined;
  private selection: "nearest" | "cycle" | undefined;
  private shotStarted: number | undefined;

  constructor(
    private readonly surface: HTMLElement,
    private readonly keyboardTarget: Window = window,
    private readonly feedback: (message: string) => void = () => {},
  ) {
    keyboardTarget.addEventListener("keydown", this.keydown);
    keyboardTarget.addEventListener("keyup", this.keyup);
    keyboardTarget.addEventListener("blur", this.clear);
    surface.addEventListener("focusout", this.focusout);
  }

  get shotCharge(): number {
    return this.shotStarted === undefined ? 0
      : clamp((this.keyboardTarget.performance.now() - this.shotStarted) / chargeMilliseconds, 0, 1);
  }

  clear = (): void => {
    this.held.clear();
    this.action = undefined;
    this.selection = undefined;
    this.shotStarted = undefined;
  };

  dispose(): void {
    this.clear();
    this.keyboardTarget.removeEventListener("keydown", this.keydown);
    this.keyboardTarget.removeEventListener("keyup", this.keyup);
    this.keyboardTarget.removeEventListener("blur", this.clear);
    this.surface.removeEventListener("focusout", this.focusout);
  }

  sample(snapshot: SessionSnapshot, config: MatchConfig): HumanInput {
    if (!this.hasMatchFocus()) this.clear();
    const action = this.action;
    const selection = this.selection;
    this.action = undefined;
    this.selection = undefined;

    const held = new Set(this.held.values());
    const x = Number(held.has("right")) - Number(held.has("left"));
    const y = Number(held.has("down")) - Number(held.has("up"));
    const length = Math.hypot(x, y);
    const movement = length > 0 ? { x: x / length, y: y / length } : neutral;
    const input: HumanInput = { movement, sprint: held.has("sprint") };
    const selected = snapshot.players.find(({ playerId }) => playerId === snapshot.selectedPlayerId);
    if (!selected) return {};

    const player = selection ? selectPlayer(snapshot, selected, selection === "cycle") : selected;
    if (selection) input.selectPlayerId = player.playerId;
    if (!action) return input;
    if (player.action) {
      this.feedback("Player is still finishing an action");
      return input;
    }

    const attack = snapshot.attackDirections[config.teams[0].id === player.teamId ? 0 : 1];
    const forward = { x: attack, y: 0 };
    const aim = length > 0 ? movement : snapshot.phase.kind === "kickoffSetup" && config.rules.directionalKickoff
      ? { x: -attack, y: 0 } : forward;
    if (snapshot.phase.kind !== "live") {
      const restartInput = restart(snapshot, config, player, action, aim);
      if (!restartInput.request && action.kind !== "defend") {
        const { phase } = snapshot;
        if ((phase.kind === "kickoffSetup" || phase.kind === "restartSetup") && phase.teamId === player.teamId) {
          this.feedback(phase.takerId !== player.playerId ? "Press Q to select the taker" : "Wait for the restart prompt");
        }
      }
      return { ...input, ...restartInput };
    }

    const { ball } = snapshot;
    if (ball.control.kind === "keeperHeld") {
      if (ball.control.playerId === player.playerId) {
        input.request = keeperRelease(snapshot, config, player, action, aim);
      } else {
        this.feedback("The keeper has the ball");
      }
      return input;
    }
    if (action.kind === "defend") {
      const ownGoal = -attack * config.pitch.lengthM / 2;
      const depth = (player.position.x - ownGoal) * attack;
      const keeperCanSave = player.definition.role === "goalkeeper" && depth >= 0 &&
        depth <= config.pitch.keeperAreaDepthM && Math.abs(player.position.y) <= config.pitch.keeperAreaWidthM / 2;
      const target = { x: ball.position.x + ball.velocity.x * 0.08, y: ball.position.y + ball.velocity.y * 0.08 };
      input.request = {
        kind: keeperCanSave ? "keeperSave" : "tackle",
        direction: direction(player.position, target, aim),
      };
      return input;
    }

    const canKick = ball.control.kind === "free" || ball.control.playerId === player.playerId;
    const ballDistance = Math.hypot(ball.position.x - player.position.x,
      ball.position.y - player.position.y, ball.position.z);
    if (!canKick || ballDistance > player.definition.abilities.physical.reachM) {
      this.feedback(canKick ? "Ball out of reach" : "Selected player doesn’t have the ball");
      return input;
    }

    if (action.kind === "shot") {
      const target = { x: attack * config.pitch.lengthM / 2, y: movement.y * config.pitch.goalWidthM * 0.32 };
      input.request = {
        kind: "kick", kickType: "shot", direction: direction(ball.position, target, forward),
        power: action.power, loft: 0.1 + action.power * 0.14,
      };
      return input;
    }
    return { ...input, ...kick(snapshot, config, player, action.kind, aim) };
  }

  private hasMatchFocus(): boolean {
    const active = this.surface.ownerDocument.activeElement;
    return !!active && this.surface.contains(active) &&
      (active === this.surface || !active.closest("button, a, input, select, textarea, [contenteditable]"));
  }

  private focusout = (event: FocusEvent): void => {
    const next = event.relatedTarget;
    if (!(next instanceof Node) || !this.surface.contains(next)) this.clear();
  };

  private keydown = (event: KeyboardEvent): void => {
    if (!this.hasMatchFocus() || event.altKey || event.ctrlKey || event.metaKey) return;
    const key = keyFor(event);
    if (!key) return;
    event.preventDefault();
    const token = tokenFor(event);
    if (event.repeat || this.held.has(token)) return;
    this.held.set(token, key);
    switch (key) {
      case "shot":
        this.shotStarted = this.keyboardTarget.performance.now();
        break;
      case "select":
        this.selection = event.shiftKey ? "cycle" : "nearest";
        break;
      case "pass":
      case "loft":
      case "defend":
        this.action = { kind: key };
    }
  };

  private keyup = (event: KeyboardEvent): void => {
    const token = tokenFor(event);
    const held = this.held.get(token);
    if (!held) return;
    this.held.delete(token);
    const focused = this.hasMatchFocus();
    if (focused) event.preventDefault();
    if (held === "shot") {
      if (focused && this.shotStarted !== undefined) this.action = { kind: "shot", power: 0.46 + 0.54 * this.shotCharge };
      this.shotStarted = undefined;
    }
  };
}
