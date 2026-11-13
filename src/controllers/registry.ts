import type { TeamAI } from "./contracts.js";

type TeamAIFactory = () => TeamAI;

export class TeamAIRegistry {
  private readonly factories = new Map<string, TeamAIFactory>();
  private readonly issued = new WeakSet<TeamAI>();

  has(id: string): boolean {
    return this.factories.has(id);
  }

  register(id: string, factory: TeamAIFactory): this {
    if (typeof id !== "string" || id.trim() === "" || typeof factory !== "function") {
      throw new TypeError("A team AI registration requires a nonempty ID and a factory");
    }
    if (this.factories.has(id)) throw new Error(`Team AI "${id}" is already registered`);
    this.factories.set(id, factory);
    return this;
  }

  create(id: string): TeamAI {
    const factory = this.factories.get(id);
    if (!factory) throw new Error(`Unknown team AI "${id}"`);
    const plugin = factory();
    if (!plugin || typeof plugin !== "object" || plugin.id !== id ||
        typeof plugin.version !== "string" || plugin.version.trim() === "" ||
        typeof plugin.reset !== "function" || typeof plugin.decide !== "function" ||
        typeof plugin.dispose !== "function") {
      throw new TypeError(`Factory for team AI "${id}" returned an invalid plugin`);
    }
    if (this.issued.has(plugin)) {
      throw new Error(`Factory for team AI "${id}" must return a fresh instance`);
    }
    this.issued.add(plugin);
    return plugin;
  }
}
