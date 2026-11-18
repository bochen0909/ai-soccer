// Optional keeper rule for experiments: KEEPER_REACH=1.5 multiplies how far every goalkeeper's save reaches
// (config.tuning.keeperReachScale). Unset, matches play exactly as before.
export function withKeeperReach(config) {
  const scale = Number(process.env.KEEPER_REACH ?? 1);
  if (scale !== 1) config.tuning = { ...config.tuning, keeperReachScale: scale };
  return config;
}
