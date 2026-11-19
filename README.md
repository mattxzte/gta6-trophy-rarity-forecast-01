# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mattxzte/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mattxzte/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Earners-over-players trophy rarity forecast with honest logit quantiles: it splits each trophy's Exophase earn rate into earners over players tracked, fits both counts on log time since release, and forecasts five quantiles per trophy on the logit scale. A pending refresh pulls most rates down as new players join before earning anything, while progression pulls story trophies back up; the spread adds the random-walk drift, the fit's own uncertainty and the size of past refresh jumps. Deterministic, Node standard library only, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
# reads every <id>.in in input/, writes <id>.out per case to out/
```

In the grader the program runs as `node main.ts` with the case at `/input` and the forecast written to `/output`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=nv10mx49&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
