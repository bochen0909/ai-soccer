import type { ActionKey, MatchEvent, MatchPhase } from "../contracts/index.js";
import { createArcadeConfig, createElevenAsideConfig } from "../presets/arcade.js";
import { MatchAudio } from "../render/audio.js";
import { PitchRenderer } from "../render/scene.js";
import { MatchSession } from "../session/match-session.js";
import type { SessionSnapshot } from "../session/contracts.js";
import { FixedStepClock } from "./fixed-step.js";
import { KeyboardControls } from "./keyboard.js";

function element<T extends HTMLElement>(id: string): T {
  const target = document.getElementById(id);
  if (!target) throw new Error(`Missing match element ${id}`);
  return target as T;
}

const surface = element("surface");
const gate = element("gate");
const gateTitle = element("gate-title");
const gateCopy = element("gate-copy");
const gateAction = element<HTMLButtonElement>("gate-action");
const pauseButton = element<HTMLButtonElement>("pause");
const soundButton = element<HTMLButtonElement>("sound");
const scoreboard = [element("home-score"), element("away-score")];
const periodLabel = element("period");
const clockLabel = element("clock");
const phaseLabel = element("phase");
const attackLabel = element("attack-direction");
const message = element("game-message");
const playerNumber = element("player-number");
const playerName = element("player-name");
const staminaMeter = element("stamina");
const shotMeter = element("shot-meter");
const shotPower = element("shot-power");
const matchTitle = element("match-title");
const matchFormat = element<HTMLSelectElement>("match-format");
const pitch = element<HTMLCanvasElement>("pitch");

let config = matchFormat.value === "full" ? createElevenAsideConfig() : createArcadeConfig();
config.rules.directionalKickoff = true;
const audio = new MatchAudio();
const input = new KeyboardControls(surface, window, (detail) => showActionFeedback(detail, timing.frame().current.tick));
let renderer = new PitchRenderer(pitch, config);
const session = new MatchSession();
const names = ["N. Kova", "M. Silva", "D. Reyes", "T. Moss", "A. Vega"];
const fullNames = ["N. Kova", "M. Silva", "D. Reyes", "I. Okafor", "R. Malik", "T. Moss",
  "L. Costa", "S. Kim", "J. Mensah", "A. Vega", "E. Sato"];
const seed = 17;

function resetSession(): void {
  const { id: teamId, players } = config.teams[0];
  const taker = players.filter((player) => player.role === "outfield").reduce((nearest, player) =>
    Math.hypot(player.formation.x, player.formation.y) < Math.hypot(nearest.formation.x, nearest.formation.y)
      ? player : nearest);
  session.reset(config, seed, { human: { teamId, playerId: taker.id } });
}

resetSession();
const timing = new FixedStepClock(session.snapshot());
type Mode = "ready" | "playing" | "paused" | "finished" | "error";
let mode: Mode = "ready";
let sound = true;
let matchMessage = "";
let messageUntil = 0;
let actionMessage = "";
let actionMessageUntil = 0;
let humanAction: ActionKey | undefined;

function isHumanAction(action: ActionKey): boolean {
  return action.source === "controller" && humanAction?.playerId === action.playerId &&
    humanAction.requestId === action.requestId;
}

function showActionFeedback(detail: string, tick: number): void {
  actionMessage = detail;
  actionMessageUntil = tick + 78;
}

function showGate(title: string, detail: string, action: string): void {
  gateTitle.textContent = title;
  gateCopy.textContent = detail;
  gateAction.textContent = action;
  gate.hidden = false;
}

function start(): void {
  audio.unlock();
  input.clear();
  timing.align();
  mode = "playing";
  gate.hidden = true;
  pauseButton.disabled = false;
  pauseButton.textContent = "Pause";
  surface.focus({ preventScroll: true });
}

function resetMatch(): void {
  resetSession();
  input.clear();
  renderer.resetBallRotation();
  timing.align(session.snapshot());
  matchMessage = "";
  messageUntil = 0;
  actionMessageUntil = 0;
  humanAction = undefined;
}

function restart(): void {
  resetMatch();
  start();
}

function showReady(): void {
  mode = "ready";
  pauseButton.disabled = true;
  pauseButton.textContent = "Pause";
  matchTitle.textContent = matchFormat.value === "full" ? "Full Match" : "Quick Match";
  showGate("Ready for kickoff?", "Take control of Ember FC.", "Kick off");
}

function pause(detail: string): void {
  if (mode !== "playing") return;
  mode = "paused";
  input.clear();
  timing.align();
  pauseButton.textContent = "Resume";
  showGate("Match paused", detail, "Resume match");
}

function finish(snapshot: SessionSnapshot): void {
  mode = "finished";
  input.clear();
  pauseButton.disabled = true;
  const [home, away] = snapshot.score;
  const result = home === away ? "Draw" : home > away ? "Ember FC wins" : "Tidal FC wins";
  showGate("Full time", `${result} · ${home}–${away}`, "Play again");
}

function receive(events: readonly MatchEvent[]): void {
  audio.play(events);
  for (const event of events) {
    if (event.kind === "goal") {
      matchMessage = event.teamId === config.teams[0].id ? "GOAL · EMBER FC" : "GOAL · TIDAL FC";
      messageUntil = event.tick + config.rules.goalPauseTicks;
    } else if (event.kind === "actionContact" && (event.outcome === "caught" || event.outcome === "parried")) {
      matchMessage = event.outcome === "caught" ? "Keeper holds it" : "Keeper parries";
      messageUntil = event.tick + 48;
    } else if (event.kind === "actionContact" && event.outcome === "missed" && isHumanAction(event.action)) {
      showActionFeedback("Missed the ball", event.tick);
    }
  }
}

function phaseText(phase: MatchPhase): string {
  switch (phase.kind) {
    case "kickoffSetup": return "Kickoff";
    case "restartSetup": return { throwIn: "Throw-in", goalKick: "Goal kick", corner: "Corner" }[phase.restart];
    case "live": return "Live";
    case "goalPause": return "Goal";
    case "halftime": return "Half time";
    case "fullTime": return "Full time";
  }
}

function matchHint(snapshot: SessionSnapshot): string {
  if (snapshot.tick < messageUntil) return matchMessage;
  const { phase, selectedPlayerId, ball } = snapshot;
  if (phase.kind === "kickoffSetup" || phase.kind === "restartSetup") {
    if (phase.teamId !== config.teams[0].id) return `${phaseText(phase)} · Tidal FC`;
    if (snapshot.tick < phase.readyTick) return `${phaseText(phase)} · Ember FC`;
    if (selectedPlayerId === phase.takerId) return phase.kind === "kickoffSetup"
      ? "J to kick off · Hold a direction to aim" : "Press J to put the ball in play";
    return "Press Q to select the taker";
  }
  if (phase.kind === "halftime") return "Half time · Teams switch ends";
  if (snapshot.tick < actionMessageUntil) return actionMessage;
  if (ball.control.kind === "keeperHeld" && ball.control.playerId === selectedPlayerId) {
    return "J to roll · L to throw";
  }
  return "";
}

function paintHud(snapshot: SessionSnapshot): void {
  for (const [index, score] of snapshot.score.entries()) scoreboard[index].textContent = String(score);
  periodLabel.textContent = snapshot.period === 1 ? "1ST" : "2ND";
  const elapsedTicks = (snapshot.period === 2 ? config.periodTicks[0] : 0) +
    config.periodTicks[snapshot.period - 1] - snapshot.remainingPeriodTicks;
  const elapsed = Math.floor(elapsedTicks / 60);
  clockLabel.textContent = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  phaseLabel.textContent = phaseText(snapshot.phase).toUpperCase();
  attackLabel.textContent = snapshot.attackDirections[0] === 1 ? "Attack →" : "← Attack";
  const selected = snapshot.players.find(({ playerId }) => playerId === snapshot.selectedPlayerId);
  const index = config.teams[0].players.findIndex(({ id }) => id === selected?.playerId);
  const fullMatch = matchFormat.value === "full";
  playerNumber.textContent = index < 0 ? "—" : String(fullMatch ? index + 1 : index === 0 ? 1 : index + 6);
  playerName.textContent = (fullMatch ? fullNames : names)[index] ?? "Ember FC";
  staminaMeter.style.transform = `scaleX(${selected?.stamina ?? 1})`;
  shotMeter.hidden = mode !== "playing" || input.shotCharge <= 0;
  shotPower.style.transform = `scaleX(${input.shotCharge})`;
  const hint = mode === "playing" ? matchHint(snapshot) : "";
  if (message.textContent !== hint) message.textContent = hint;
  message.hidden = hint === "";
}

function draw(now: number): void {
  let frame = timing.frame();
  if (mode === "playing") {
    try {
      frame = timing.advance(now, () => {
        const previous = timing.frame().current;
        const command = input.sample(previous, config);
        const result = session.step(command);
        const snapshot = session.snapshot();
        if (command.request && snapshot.selectedPlayerId) {
          const playerId = snapshot.selectedPlayerId;
          const submitted = result.frame.actions.find((action) => action.playerId === playerId)?.request;
          if (submitted) {
            humanAction = { source: "controller", playerId, requestId: submitted.id };
            actionMessageUntil = 0;
          }
        }
        receive(result.events);
        const actionResult = result.actionResults.find(({ action, status }) => isHumanAction(action) && status !== "accepted");
        if (actionResult?.status === "rejected") {
          const detail = actionResult.reason === "busy" ? "Player is still finishing an action" :
            actionResult.reason === "ballUnavailable" ? "Ball unavailable for that action" : "Action unavailable right now";
          showActionFeedback(detail, snapshot.tick);
        } else if (actionResult?.status === "interrupted" && actionResult.reason === "ballUnavailable") {
          showActionFeedback("Lost the ball before contact", snapshot.tick);
        }
        if (actionResult) humanAction = undefined;
        const placed = result.events.some((event) => event.kind === "phaseChanged" &&
          (event.phase.kind === "restartSetup" || event.phase.kind === "kickoffSetup"));
        if (placed) renderer.resetBallRotation();
        else renderer.advanceBallRotation(previous, snapshot);
        return { snapshot, align: placed, finished: result.status === "finished" };
      });
      if (frame.finished) finish(frame.current);
      else if (frame.pauseReason) pause("The match paused while the browser was busy.");
    } catch (error) {
      console.error(error);
      mode = "error";
      input.clear();
      timing.align();
      pauseButton.disabled = true;
      showGate("Match interrupted", "Start a new match to continue.", "New match");
      frame = timing.frame();
    }
  }
  const pass = mode === "playing" && matchFormat.value === "full"
    ? input.previewGroundPass(frame.current, config) : undefined;
  renderer.draw(frame.previous, frame.current, frame.alpha, pass?.receiverId);
  paintHud(frame.current);
  requestAnimationFrame(draw);
}

gateAction.addEventListener("click", () => mode === "finished" || mode === "error" ? restart() : start());
matchFormat.addEventListener("change", () => {
  config = matchFormat.value === "full" ? createElevenAsideConfig() : createArcadeConfig();
  config.rules.directionalKickoff = true;
  renderer = new PitchRenderer(pitch, config);
  resetMatch();
  showReady();
});
element("restart").addEventListener("click", restart);
pauseButton.addEventListener("click", () => mode === "playing" ? pause("Resume whenever you’re ready.") : start());
soundButton.addEventListener("click", () => {
  sound = !sound;
  audio.setEnabled(sound);
  soundButton.textContent = sound ? "Sound on" : "Sound off";
  soundButton.setAttribute("aria-pressed", String(sound));
  if (mode === "playing") surface.focus({ preventScroll: true });
});
window.addEventListener("keydown", (event) => {
  if (event.code !== "Escape" || event.repeat) return;
  if (mode === "playing") pause("Resume whenever you’re ready.");
  else if (mode === "paused" && surface.contains(document.activeElement)) start();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) pause("Resume to return to the pitch.");
});
window.addEventListener("blur", () => pause("Resume to return to the pitch."));
showReady();
requestAnimationFrame(draw);
