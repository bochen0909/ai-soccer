import type { ActionKey, MatchEvent, MatchPhase } from "../contracts/index.js";
import { createArcadeConfig, createElevenAsideConfig } from "../presets/arcade.js";
import { isAIProfileId, profileSupports, profileTeamOptions } from "../presets/ai.js";
import type { AIProfileId } from "../presets/ai.js";
import { MatchAudio } from "../render/audio.js";
import { PitchRenderer } from "../render/scene.js";
import { registerNeuralPolicy } from "../controllers/neural.js";
import type { PolicyWeights } from "../controllers/neural.js";
import { createDefaultAIRegistry, MatchSession } from "../session/match-session.js";
import type { HumanInput, SessionSnapshot } from "../session/contracts.js";
import { createDrill } from "../training/drills.js";
import type { DrillKind } from "../training/drills.js";
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
const drillSelect = element<HTMLSelectElement>("drill");
const homeControl = element<HTMLSelectElement>("home-control");
const awayControl = element<HTMLSelectElement>("away-control");
const homeTag = element("home-tag");
const awayTag = element("away-tag");
const playerCard = element("player-card");
const controlsPanel = element("controls");
const pitch = element<HTMLCanvasElement>("pitch");

let config = matchFormat.value === "full" ? createElevenAsideConfig() : createArcadeConfig();
config.rules.directionalKickoff = true;
const audio = new MatchAudio();
const input = new KeyboardControls(surface, window, (detail) => showActionFeedback(detail, timing.frame().current.tick));
let renderer = new PitchRenderer(pitch, config);
const registry = createDefaultAIRegistry();
let trainedReady = false;
try {
  // The trained policy is a 600 KB file. Without it the page still works, just without the Trained option.
  // Add ?model=candidates/role_gen13 (or anchors/gen12-role_gen12) to the address to play against another policy.
  const requested = new URLSearchParams(location.search).get("model");
  const path = requested !== null && /^(anchors|candidates)\/[\w.-]+$/.test(requested)
    ? `/models/${requested}.json` : "/models/trained-5v5.json";
  const response = await fetch(path);
  if (response.ok) {
    registerNeuralPolicy(registry, (await response.json()) as PolicyWeights);
    trainedReady = true;
  }
} catch (error) {
  console.warn("Trained policy unavailable", error);
}
const session = new MatchSession(registry);
const names = ["N. Kova", "M. Silva", "D. Reyes", "T. Moss", "A. Vega"];
const fullNames = ["N. Kova", "M. Silva", "D. Reyes", "I. Okafor", "R. Malik", "T. Moss",
  "L. Costa", "S. Kim", "J. Mensah", "A. Vega", "E. Sato"];
/** Every game gets a fresh seed. Add ?seed=N to the page address to replay a particular game. */
const pinnedSeed = new URLSearchParams(location.search).get("seed");
const nextSeed = (): number => pinnedSeed !== null && Number.isInteger(Number(pinnedSeed))
  ? Number(pinnedSeed) >>> 0 : Math.floor(Math.random() * 0x100000000);

/** A drill plays from a built position, so it is watch-only and only exists in the 5-a-side format. */
function activeDrill(): DrillKind | undefined {
  return matchFormat.value === "quick" && drillSelect.value !== "" ? drillSelect.value as DrillKind : undefined;
}
const DRILL_MAX_TICKS = 900;
let drillStartTick = 0;

const requestedDrill = new URLSearchParams(location.search).get("drill");
if (requestedDrill !== null && Array.from(drillSelect.options).some((option) => option.value === requestedDrill)) {
  drillSelect.value = requestedDrill;
}

function humanControlled(): boolean {
  return homeControl.value === "human" && activeDrill() === undefined;
}

function profileOf(select: HTMLSelectElement): AIProfileId {
  return isAIProfileId(select.value) ? select.value : "default";
}

function resetSession(): void {
  const { id: teamId, players } = config.teams[0];
  const taker = players.filter((player) => player.role === "outfield").reduce((nearest, player) =>
    Math.hypot(player.formation.x, player.formation.y) < Math.hypot(nearest.formation.x, nearest.formation.y)
      ? player : nearest);
  // The human's teammates always use the default profile.
  const teams = {
    [config.teams[0].id]: profileTeamOptions(humanControlled() ? "default" : profileOf(homeControl)),
    [config.teams[1].id]: profileTeamOptions(profileOf(awayControl)),
  };
  const seed = nextSeed();
  const drill = activeDrill();
  const scenario = drill === undefined ? undefined : createDrill(drill, config, 0, seed);
  session.reset(config, seed, humanControlled() ? { human: { teamId, playerId: taker.id }, teams } : { teams }, scenario);
  drillStartTick = session.snapshot().tick;
}

/** Tuned profiles only apply to formats they were tuned for. Falls back to the default profile otherwise. */
function syncControlOptions(): void {
  const playersPerSide = config.teams[0].players.length;
  drillSelect.addEventListener("change", () => {
  syncControlOptions();
  resetMatch();
  showReady();
});
for (const select of [homeControl, awayControl]) {
    for (const option of Array.from(select.options)) {
      if (isAIProfileId(option.value)) {
        option.disabled = !profileSupports(option.value, playersPerSide) || (option.value === "trained" && !trainedReady);
      }
    }
    if (select.selectedOptions[0]?.disabled) select.value = "default";
  }
  const drilling = activeDrill() !== undefined;
  drillSelect.disabled = matchFormat.value !== "quick";
  if (matchFormat.value !== "quick") drillSelect.value = "";
  const humanOption = Array.from(homeControl.options).find((option) => option.value === "human");
  if (humanOption) humanOption.disabled = drilling;
  if (drilling && homeControl.value === "human") homeControl.value = trainedReady ? "trained" : "tuned";
  homeTag.textContent = humanControlled() ? "YOU" : "CPU";
  awayTag.textContent = "CPU";
  playerCard.hidden = !humanControlled();
  controlsPanel.hidden = !humanControlled();
}

syncControlOptions();
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
  const drill = activeDrill();
  matchTitle.textContent = drill !== undefined ? `Drill · ${drillSelect.selectedOptions[0]?.textContent?.replace("Drill · ", "") ?? drill}` :
    matchFormat.value === "full" ? "Full Match" : "Quick Match";
  if (drill !== undefined) showGate("Ready?", "Watch the drill. It ends on a goal, a dead ball or after 15 seconds.", "Start drill");
  else showGate("Ready for kickoff?",
    humanControlled() ? "Take control of Ember FC." : "Watch Ember FC play Tidal FC.", "Kick off");
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
  if (activeDrill() !== undefined) {
    showGate("Drill complete", home > away ? "Ember FC scored" : away > home ? "Tidal FC scored" : "No goal", "Next drill");
    return;
  }
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
  if (!humanControlled()) {
    if (phase.kind === "kickoffSetup" || phase.kind === "restartSetup") {
      return `${phaseText(phase)} · ${phase.teamId === config.teams[0].id ? "Ember FC" : "Tidal FC"}`;
    }
    return phase.kind === "halftime" ? "Half time · Teams switch ends" : "";
  }
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
        const command: HumanInput = humanControlled() ? input.sample(previous, config) : {};
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
        const drillOver = activeDrill() !== undefined && (snapshot.tick - drillStartTick >= DRILL_MAX_TICKS ||
          result.events.some((event) => event.kind === "goal" || event.kind === "ballExit"));
        return { snapshot, align: placed, finished: result.status === "finished" || drillOver };
      });
      if (frame.finished) finish(frame.current);
      else if (frame.pauseReason) {
        // A watched CPU match has nothing to protect, so resume the clock instead of gating.
        if (humanControlled()) pause("The match paused while the browser was busy.");
        else timing.align();
      }
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
  syncControlOptions();
  resetMatch();
  showReady();
});
for (const select of [homeControl, awayControl]) {
  select.addEventListener("change", () => {
    syncControlOptions();
    resetMatch();
    showReady();
  });
}
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
  if (document.hidden && humanControlled()) pause("Resume to return to the pitch.");
});
window.addEventListener("blur", () => { if (humanControlled()) pause("Resume to return to the pitch."); });
showReady();
requestAnimationFrame(draw);
