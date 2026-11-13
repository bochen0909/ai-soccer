/** A 32-bit seeded stream. Every uint32, including zero, is a valid seed. */
export function randomUnit(state: { rngState: number }): number {
  state.rngState = (state.rngState + 0x6d2b79f5) >>> 0;
  let mixed = state.rngState;
  mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
  mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
  return ((mixed ^ (mixed >>> 14)) >>> 0) / 0x100000000;
}

/** Who takes a kickoff: the usual team, or either team with equal chance when the match uses random kickoffs. */
export function kickoffTeamId(
  state: { rngState: number; config: { rules: { randomKickoff?: boolean }; teams: readonly { id: string }[] } },
  usual: string,
): string {
  return state.config.rules.randomKickoff ? state.config.teams[randomUnit(state) < 0.5 ? 0 : 1].id : usual;
}
