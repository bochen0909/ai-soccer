import type { MatchConfig, PlayerState, Vec2, Vec3 } from "../contracts/index.js";
import type { SessionSnapshot } from "../session/contracts.js";

type ViewPlayer = SessionSnapshot["players"][number];
type Project = (x: number, y: number, z?: number) => Vec2;

interface Kit {
  light: string;
  main: string;
  dark: string;
  shorts: string;
}

const homeKit = { light: "#ffb77c", main: "#f17845", dark: "#a83f2e", shorts: "#30252c" };
const awayKit = { light: "#b0f7ff", main: "#58cce4", dark: "#247c9a", shorts: "#183447" };
const keeperKits = [
  { light: "#ffeea7", main: "#e9c465", dark: "#8e702f", shorts: "#443929" },
  { light: "#dbcaff", main: "#ad8ce8", dark: "#624395", shorts: "#382d53" },
];
const skinColors = ["#d99a70", "#b77851", "#e6af87", "#8c5741", "#c68862"];

function lerp(from: number, to: number, alpha: number): number {
  return from + (to - from) * alpha;
}

function lerpFacing(from: number, to: number, alpha: number): number {
  const difference = Math.atan2(Math.sin(to - from), Math.cos(to - from));
  return from + difference * alpha;
}

function path(ctx: CanvasRenderingContext2D, points: readonly Vec2[], close = false): void {
  ctx.beginPath();
  points.forEach(({ x, y }, index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  if (close) ctx.closePath();
}

function polygon(ctx: CanvasRenderingContext2D, points: readonly Vec2[], color: string): void {
  ctx.fillStyle = color;
  path(ctx, points, true);
  ctx.fill();
}

function line(ctx: CanvasRenderingContext2D, points: readonly Vec2[], color: string, width: number): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  path(ctx, points);
  ctx.stroke();
}

function ellipse(
  ctx: CanvasRenderingContext2D, point: Vec2, radiusX: number, radiusY: number, color: string,
): void {
  ctx.beginPath();
  ctx.ellipse(point.x, point.y, radiusX, radiusY, 0, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function noise(index: number): number {
  return ((Math.imul(index ^ 0x792ae5d3, 0x45d9f3b) >>> 8) & 0xffff) / 0x10000;
}

/** Projects the engine's pitch coordinates into a fixed elevated sideline view. */
export class PitchRenderer {
  private readonly context: CanvasRenderingContext2D;
  private readonly background = document.createElement("canvas");
  private width = 0;
  private height = 0;
  private pixelRatio = 0;
  private scale = 1;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly config: MatchConfig) {
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) throw new Error("Canvas rendering is unavailable");
    this.context = context;
  }

  private readonly project: Project = (x, y, z = 0) => {
    const perspective = 1 + y / this.config.pitch.widthM * 0.16;
    return {
      x: this.width * 0.5 + x * perspective * this.scale,
      y: this.height * 0.51 + (y * 0.68 - z * 0.8) * this.scale,
    };
  };

  draw(previous: SessionSnapshot, current: SessionSnapshot, alpha: number): void {
    if (!this.resize()) return;
    const ctx = this.context;
    ctx.drawImage(this.background, 0, 0, this.width, this.height);

    const oldPlayers = new Map(previous.players.map((player) => [player.playerId, player]));
    const players = current.players.map((player) => {
      const old = oldPlayers.get(player.playerId) ?? player;
      return {
        ...player,
        position: {
          x: lerp(old.position.x, player.position.x, alpha),
          y: lerp(old.position.y, player.position.y, alpha),
        },
        facingRad: lerpFacing(old.facingRad, player.facingRad, alpha),
      };
    });
    const ball = {
      x: lerp(previous.ball.position.x, current.ball.position.x, alpha),
      y: lerp(previous.ball.position.y, current.ball.position.y, alpha),
      z: lerp(previous.ball.position.z, current.ball.position.z, alpha),
    };
    const time = lerp(previous.tick, current.tick, alpha) / 60;
    for (const player of players) this.drawPlayerShadow(player);
    this.drawBallShadow(ball);
    const selected = players.find(({ playerId }) => playerId === current.selectedPlayerId);
    if (selected) this.drawSelection(selected, time);
    const { control } = current.ball;
    if (control.kind !== "free") {
      const holder = players.find(({ playerId }) => playerId === control.playerId);
      if (holder) {
        const point = this.project(holder.position.x, holder.position.y);
        const size = Math.max(5, this.scale * 0.52);
        ellipse(ctx, point, size, size * 0.53, "#102018");
        ellipse(ctx, point, size - 1.4, size * 0.53 - 0.9,
          holder.teamId === this.config.teams[0].id ? "#ffae60" : "#67deff");
      }
    }
    const sorted = players.sort((first, second) => first.position.y - second.position.y);
    for (const player of sorted) this.drawPlayer(player, time);
    this.drawBall(ball);
  }

  private resize(): boolean {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (width <= 0 || height <= 0) return false;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (width === this.width && height === this.height && ratio === this.pixelRatio) return true;
    this.width = width;
    this.height = height;
    this.pixelRatio = ratio;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const { lengthM, widthM } = this.config.pitch;
    this.scale = Math.min(width / (lengthM * 1.085 + 10), height / (widthM * 0.68 + 7.2));
    this.background.width = this.canvas.width;
    this.background.height = this.canvas.height;
    const context = this.background.getContext("2d");
    if (context) {
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.drawBackground(context);
    }
    return true;
  }

  private rect(left: number, right: number, back: number, front: number, z = 0): Vec2[] {
    return [
      this.project(left, back, z), this.project(right, back, z),
      this.project(right, front, z), this.project(left, front, z),
    ];
  }

  private drawBackground(ctx: CanvasRenderingContext2D): void {
    const { lengthM, widthM } = this.config.pitch;
    const x = lengthM / 2;
    const y = widthM / 2;
    const stadium = ctx.createRadialGradient(this.width / 2, this.height * 0.42, this.height * 0.1,
      this.width / 2, this.height * 0.42, this.width * 0.62);
    stadium.addColorStop(0, "#20353c");
    stadium.addColorStop(1, "#081419");
    ctx.fillStyle = stadium;
    ctx.fillRect(0, 0, this.width, this.height);

    this.drawStand(ctx, y);
    polygon(ctx, this.rect(-x - 3.8, x + 3.8, -y - 3.6, y + 3.7), "#122625");
    polygon(ctx, this.rect(-x - 2.3, x + 2.3, -y - 2.2, y + 2.2), "#294139");
    polygon(ctx, this.rect(-x - 1.6, x + 1.6, -y - 1.5, y + 1.5), "#22513b");

    const stripes = 12;
    for (let index = 0; index < stripes; index++) {
      const left = -x + lengthM * index / stripes;
      const right = -x + lengthM * (index + 1) / stripes;
      polygon(ctx, this.rect(left, right, -y, y), index % 2 === 0 ? "#26734b" : "#226744");
    }
    const grassLight = ctx.createLinearGradient(0, this.project(0, -y).y, 0, this.project(0, y).y);
    grassLight.addColorStop(0, "rgba(119, 213, 158, 0.08)");
    grassLight.addColorStop(0.57, "rgba(104, 196, 145, 0.03)");
    grassLight.addColorStop(1, "rgba(4, 27, 22, 0.18)");
    ctx.fillStyle = grassLight;
    path(ctx, this.rect(-x, x, -y, y), true);
    ctx.fill();

    // A fine static grain keeps the turf textured without introducing frame noise.
    for (let index = 0; index < 4200; index++) {
      const point = this.project(lerp(-x, x, noise(index * 3)), lerp(-y, y, noise(index * 3 + 1)));
      ctx.fillStyle = index % 2 ? "rgba(186, 234, 178, 0.055)" : "rgba(4, 35, 24, 0.09)";
      ctx.fillRect(point.x, point.y, 0.5 + noise(index * 3 + 2), 0.6);
    }
    this.drawMarkings(ctx);
    this.drawGoal(ctx, -1);
    this.drawGoal(ctx, 1);
  }

  private drawStand(ctx: CanvasRenderingContext2D, halfWidth: number): void {
    const front = this.project(0, -halfWidth - 2.4).y;
    const length = this.width * 0.94;
    const left = (this.width - length) / 2;
    const back = front - this.scale * 3.1;
    const wall = ctx.createLinearGradient(0, back, 0, front);
    wall.addColorStop(0, "#14272e");
    wall.addColorStop(1, "#243941");
    ctx.fillStyle = wall;
    ctx.fillRect(left, back, length, front - back);
    for (let row = 0; row < 4; row++) {
      const rowY = back + (row + 0.25) * (front - back) / 4;
      line(ctx, [{ x: left, y: rowY + 3 }, { x: left + length, y: rowY + 3 }], "#14272d", 1);
      for (let seat = 0; seat < 168; seat++) {
        const seatX = left + (seat + 0.5) * length / 168;
        if (seat % 21 < 2 || noise(row * 168 + seat) > 0.82) continue;
        const colors = ["#34514f", "#3d5963", "#51564e", "#2d424a", "#425b62"];
        ctx.fillStyle = colors[Math.floor(noise(seat * 17 + row * 19) * colors.length)];
        ctx.fillRect(seatX, rowY + noise(seat + row) * 2, Math.max(1, length / 290), 2);
      }
    }
    ctx.fillStyle = "#536962";
    ctx.fillRect(left, front - 1, length, 1);
    ctx.fillStyle = "#0c1e23";
    ctx.fillRect(left, front, length, Math.max(2, this.scale * 0.45));
    for (let panel = 0; panel < 16; panel++) {
      const panelWidth = length / 16;
      ctx.fillStyle = panel % 4 === 0 ? "#22443f" : "#253c43";
      ctx.fillRect(left + panel * panelWidth + 1, front + 1, panelWidth - 2, Math.max(1, this.scale * 0.27));
    }
  }

  private drawMarkings(ctx: CanvasRenderingContext2D): void {
    const pitch = this.config.pitch;
    const x = pitch.lengthM / 2;
    const y = pitch.widthM / 2;
    const white = "rgba(220, 249, 223, 0.83)";
    const weight = Math.max(0.8, this.scale * 0.075);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    const marking = (points: readonly Vec2[], closed = false) => {
      path(ctx, points, closed);
      ctx.strokeStyle = white;
      ctx.lineWidth = weight;
      ctx.stroke();
    };
    marking(this.rect(-x, x, -y, y), true);
    marking([this.project(0, -y), this.project(0, y)]);
    const circle = (cx: number, cy: number, radius: number, start: number, end: number, segments: number) =>
      Array.from({ length: segments + 1 }, (_, i) => {
        const angle = lerp(start, end, i / segments);
        return this.project(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
      });
    marking(circle(0, 0, pitch.centerCircleRadiusM, 0, Math.PI * 2, 50), true);
    ellipse(ctx, this.project(0, 0), weight * 1.4, weight, white);
    for (const side of [-1, 1]) {
      const box = (depth: number, width: number) => {
        const inside = side * (x - depth);
        marking([this.project(side * x, -width / 2), this.project(inside, -width / 2),
          this.project(inside, width / 2), this.project(side * x, width / 2)]);
      };
      box(pitch.keeperAreaDepthM, pitch.keeperAreaWidthM);
      box(pitch.goalAreaDepthM, pitch.goalAreaWidthM);
      ellipse(ctx, this.project(side * (x - pitch.keeperAreaDepthM * 0.76), 0), weight, weight * 0.7, white);
      for (const sideline of [-1, 1]) {
        const begin = side === 1 ? (sideline === 1 ? Math.PI : Math.PI / 2)
          : (sideline === 1 ? -Math.PI / 2 : 0);
        marking(circle(side * x, sideline * y, 0.85, begin, begin + Math.PI / 2, 10));
      }
    }
  }

  private drawGoal(ctx: CanvasRenderingContext2D, side: number): void {
    const { lengthM, goalWidthM, goalHeightM } = this.config.pitch;
    const x = side * lengthM / 2;
    const back = x + side * 1.65;
    const y = goalWidthM / 2;
    const roof = goalHeightM;
    polygon(ctx, this.rect(Math.min(x, back), Math.max(x, back), -y, y), "#19392f");
    polygon(ctx, [this.project(x, -y, roof), this.project(back, -y, roof),
      this.project(back, y, roof), this.project(x, y, roof)], "rgba(194, 224, 212, .12)");
    const net = "rgba(192, 224, 214, .44)";
    const mesh = Math.max(0.4, this.scale * 0.033);
    for (let index = 0; index <= 12; index++) {
      const at = lerp(-y, y, index / 12);
      line(ctx, [this.project(x, at, roof), this.project(back, at, roof), this.project(back, at)], net, mesh);
    }
    for (let index = 0; index <= 5; index++) {
      const at = roof * index / 5;
      line(ctx, [this.project(x, -y, at), this.project(back, -y, at),
        this.project(back, y, at), this.project(x, y, at)], net, mesh);
    }
    for (let index = 1; index < 4; index++) {
      const at = lerp(x, back, index / 4);
      line(ctx, [this.project(at, -y), this.project(at, -y, roof),
        this.project(at, y, roof), this.project(at, y)], net, mesh);
    }
    const frame = [this.project(x, -y), this.project(x, -y, roof),
      this.project(x, y, roof), this.project(x, y)];
    line(ctx, frame, "#183b3a", Math.max(2, this.scale * 0.23));
    line(ctx, frame, "#e9f7ee", Math.max(1.15, this.scale * 0.14));
  }

  private drawPlayerShadow(player: PlayerState): void {
    const { x, y } = player.position;
    const size = player.definition.radiusM * this.scale;
    ellipse(this.context, this.project(x + 0.09, y + 0.09), size * 1.58, size * 0.84, "rgba(5, 28, 22, .29)");
    ellipse(this.context, this.project(x, y), size * 0.87, size * 0.49, "rgba(3, 24, 19, .28)");
  }

  private drawSelection(player: PlayerState, time: number): void {
    const ctx = this.context;
    const { x, y } = player.position;
    const center = this.project(x, y);
    const radius = Math.max(5, this.scale * 0.59);
    ctx.beginPath();
    ctx.ellipse(center.x, center.y, radius, radius * 0.62, 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 185, 112, 0.19)";
    ctx.fill();
    ctx.strokeStyle = "#ffd3a0";
    ctx.lineWidth = 1.4;
    ctx.stroke();

    const head = this.project(x, y, player.definition.bodyHeightM + 0.43);
    const markerY = head.y + 1.3 * Math.sin(time * 4);
    const size = Math.max(3.5, this.scale * 0.28);
    polygon(ctx, [
      { x: head.x - size - 1, y: markerY - 1 }, { x: head.x + size + 1, y: markerY - 1 },
      { x: head.x, y: markerY + size + 1 },
    ], "#18251f");
    polygon(ctx, [
      { x: head.x - size, y: markerY }, { x: head.x + size, y: markerY },
      { x: head.x, y: markerY + size },
    ], "#ffbb84");
  }

  private drawPlayer(player: ViewPlayer, time: number): void {
    const ctx = this.context;
    const teamIndex = Number(player.teamId !== this.config.teams[0].id);
    const kit = player.definition.role === "goalkeeper" ? keeperKits[teamIndex] : teamIndex === 0 ? homeKit : awayKit;
    const hash = Array.from(player.playerId).reduce((sum, char) => sum + char.charCodeAt(0), 0);
    const skin = skinColors[hash % skinColors.length];
    const height = player.definition.bodyHeightM;
    const width = player.definition.radiusM;
    const { x, y } = player.position;
    const facing = player.facingRad;
    const front = { x: Math.cos(facing), y: Math.sin(facing) };
    const right = { x: -front.y, y: front.x };
    const speed = Math.hypot(player.velocity.x, player.velocity.y);
    const stride = Math.min(1, speed / 4.3);
    const phase = time * (9 + 4 * stride) + hash;
    const swinging = Math.sin(phase) * stride;
    const bob = height * Math.abs(Math.sin(phase)) * stride * 0.03;
    const windup = player.action?.phase === "windup" ? 1 : 0;
    const active = player.action?.phase === "active" ? 1 : 0;
    const contactFollowThrough = player.action?.phase === "recovery"
      ? Math.min(1, Math.max(0, 1 - (time * 60 - player.action.contactTick) / 9)) : 0;
    const reach = player.action?.kind === "keeperSave" ? active + windup * 0.5 : 0;
    const overhead = player.action?.kind === "throwIn" || player.action?.kind === "keeperRelease"
      ? windup + contactFollowThrough * 0.65 : 0;
    const kick = player.action?.kind === "kick" || player.action?.kind === "tackle"
      ? (contactFollowThrough + active * 0.7 - windup * 0.42) * 0.52 : 0;
    const at = (side: number, depth: number, z: number) =>
      this.project(x + right.x * side + front.x * depth, y + right.y * side + front.y * depth, z * height + bob);
    const unit = Math.max(0.55, this.scale * width);
    const limb = (points: readonly Vec2[], color: string, thickness: number) => {
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      line(ctx, points, "#132724", thickness + Math.max(0.9, unit * 0.24));
      line(ctx, points, color, thickness);
    };
    const leg = (side: number) => {
      const swing = side * swinging * height * 0.26 + (side === 1 ? kick : 0);
      const hip = at(side * width * 0.45, 0, 0.51);
      const knee = at(side * width * 0.47, swing * 0.52 + 0.04, 0.27 + Math.max(0, swing) * 0.12);
      const foot = at(side * width * 0.55, swing, 0.035 + Math.abs(swing) * 0.16);
      limb([hip, knee], kit.shorts, unit * 0.79);
      limb([knee, foot], kit.dark, unit * 0.55);
      limb([foot, { x: foot.x + front.x * unit * 0.53, y: foot.y + front.y * unit * 0.32 }],
        "#e1e9d9", unit * 0.45);
    };
    const arm = (side: number) => {
      const shoulder = at(side * width * 0.85, 0.015, 0.77);
      const elbow = at(side * width * (1.12 + reach * 0.85), -side * swinging * 0.18,
        0.63 + reach * 0.13 + overhead * 0.24);
      const hand = at(side * width * (1.15 + reach * 1.3), -side * swinging * 0.27 + reach * 0.13,
        0.53 + reach * 0.23 + overhead * 0.55);
      limb([shoulder, elbow], side === 1 ? kit.main : kit.dark, unit * 0.64);
      limb([elbow, hand], reach ? "#e7eee1" : skin, unit * 0.48);
    };

    const farSide = right.y > 0 ? -1 : 1;
    leg(farSide);
    leg(-farSide);
    arm(farSide);
    this.drawTorso(at, kit, width, unit, front.y, farSide);
    arm(-farSide);

    const head = at(0, 0.035, 0.94);
    const headRadius = Math.max(1.5, unit * 0.66);
    ellipse(ctx, head, headRadius + 0.55, headRadius * 1.05 + 0.55, "#162b28");
    ellipse(ctx, head, headRadius, headRadius * 1.05, skin);
    ellipse(ctx, { x: head.x - front.x * 0.42, y: head.y - headRadius * 0.49 },
      headRadius * 0.88, headRadius * 0.63, hash % 4 === 0 ? "#5a4138" : "#2a2825");
  }

  private drawTorso(
    at: (side: number, depth: number, z: number) => Vec2, kit: Kit, width: number, unit: number,
    frontY: number, farSide: number,
  ): void {
    const ctx = this.context;
    const chest = width * 0.42;
    const visibleFace = frontY >= 0 ? chest : -chest;
    const nearSide = -farSide;
    const side = [
      at(nearSide * width * 0.9, -chest, 0.8), at(nearSide * width * 0.9, chest, 0.8),
      at(nearSide * width * 0.68, chest, 0.48), at(nearSide * width * 0.68, -chest, 0.48),
    ];
    const outline = [
      at(-width * 0.9, visibleFace, 0.8), at(width * 0.9, visibleFace, 0.8),
      at(width * 0.68, visibleFace, 0.48), at(-width * 0.68, visibleFace, 0.48),
    ];
    ctx.strokeStyle = "#182c25";
    ctx.lineWidth = Math.max(1.1, unit * 0.39);
    for (const face of [side, outline]) {
      path(ctx, face, true);
      ctx.stroke();
    }
    polygon(ctx, side, kit.dark);
    polygon(ctx, outline, kit.main);
    polygon(ctx, [outline[0], outline[1], at(width * 0.82, -visibleFace, 0.82),
      at(-width * 0.82, -visibleFace, 0.82)], kit.light);
    polygon(ctx, [at(-width * 0.22, visibleFace, 0.78), at(width * 0.22, visibleFace, 0.78),
      at(width * 0.16, visibleFace, 0.52), at(-width * 0.16, visibleFace, 0.52)], kit.dark);
  }

  private drawBallShadow(ball: Vec3): void {
    const shadowRadius = Math.max(2.3, this.scale * (0.19 + ball.z * 0.08));
    ellipse(this.context, this.project(ball.x + 0.12, ball.y + 0.1),
      shadowRadius, shadowRadius * 0.56, "rgba(5, 24, 22, .45)");
  }

  private drawBall(ball: Vec3): void {
    const ctx = this.context;
    const center = this.project(ball.x, ball.y, ball.z);
    const radius = Math.max(3, this.scale * this.config.ball.radiusM * 1.62);
    ellipse(ctx, center, radius + 2, radius + 2, "rgba(255, 240, 196, .48)");
    ellipse(ctx, center, radius + 1.1, radius + 1.1, "#102019");
    ellipse(ctx, center, radius, radius, "#fff4db");
    ellipse(ctx, { x: center.x + radius * 0.29, y: center.y + radius * 0.22 },
      radius * 0.34, radius * 0.32, "#303c37");
    ellipse(ctx, { x: center.x - radius * 0.34, y: center.y - radius * 0.27 },
      radius * 0.19, radius * 0.17, "#758078");
  }
}
