/** A 32-bit seeded stream. Every uint32, including zero, is a valid seed. */
export function randomUnit(state: { rngState: number }): number {
  state.rngState = (state.rngState + 0x6d2b79f5) >>> 0;
  let mixed = state.rngState;
  mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
  mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
  return ((mixed ^ (mixed >>> 14)) >>> 0) / 0x100000000;
}
