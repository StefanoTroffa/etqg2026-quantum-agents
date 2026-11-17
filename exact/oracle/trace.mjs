// Oracle traces: the referee's ground truth for the Python port of the rules.
//
// Plays random games with the rule functions of the organisers' harness (workshop_resources/local-tiqtaqtoe-server.js)
// on the published @quantumplayed/quantum-game-engine. At every non-terminal state visited it expands each legal move
// into all of its measurement outcomes, by forcing the engine's measureTarget to each outcome in turn, and writes one
// JSON line: the state, the legal moves, and per move the outcomes with their probabilities. Squares are zero-based.
//
// Usage (from exact/):
//   node oracle/trace.mjs --engine <engine ESM entry file> --harness <harness file> \
//     --quantumness Q --decoherence D --games N --seed S > results/data/oracle/q{Q}-d{D}.jsonl
// JSONL goes to stdout; the record and (state, move) counts go to stderr.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const HARNESS_CUT = "// 3. Local Random AI implementation"; // the harness's own match loop and CLI start here
const MAX_PLIES = 60;
const MIN_PROBABILITY = 1e-12;

// The harness rule functions, evaluated from its source without modifying or copying the file.
function loadHarness(file, engine) {
  const lines = readFileSync(file, "utf8").split("\n");
  const cut = lines.findIndex((line) => line.trim() === HARNESS_CUT);
  if (cut === -1) throw new Error(`${file}: line "${HARNESS_CUT}" not found`);
  const text = lines.slice(0, cut).join("\n").replace(/^#!.*/, "");
  const tail = `
QuantumEntity = engine.QuantumEntity;
Operations = engine.Operations;
EntanglementEngine = engine.EntanglementEngine;
JointQuantumSystem = engine.JointQuantumSystem;
return { createFreshRound, getAvailableMoves, doMove };`;
  return new Function("require", "engine", `${text}\n${tail}`)(createRequire(import.meta.url), engine);
}

// The engine samples a measurement with r uniform on [0, 1): the first outcome k, in ascending order, with r <= C_k
// (C_k the cumulative unnormalised mass through k), else the last outcome, which so absorbs mass lost to pruning.
// Outcome k therefore has probability c(C_k) - c(C_(k-1)), the last 1 - c(C_(last-1)), with c(x) = min(x, 1).
function samplerOptions(masses) {
  const outcomes = Object.keys(masses).map(Number).sort((a, b) => a - b);
  let cumulative = 0;
  const options = outcomes.map((outcome, k) => {
    const before = Math.min(cumulative, 1);
    cumulative += masses[outcome];
    const p = k === outcomes.length - 1 ? 1 - before : Math.min(cumulative, 1) - before;
    return { outcome, p };
  });
  return options.filter(({ p }) => p > MIN_PROBABILITY);
}

// The doMove being replayed: `path` is the option to take at each measurement call (past its end: option 0), `taken`
// and `counts` log the option taken and the options available at each call, `p` the product of the probabilities.
let replay = null;

// Makes every measurement follow `replay.path` instead of sampling.
function forceMeasurements(JointQuantumSystem) {
  const measureTarget = JointQuantumSystem.prototype.measureTarget;
  JointQuantumSystem.prototype.measureTarget = function (entity) {
    const options = samplerOptions(this.getTargetProbabilities(entity));
    const index = replay.path[replay.taken.length] ?? 0;
    replay.taken.push(index);
    replay.counts.push(options.length);
    replay.p *= options[index].p;
    return measureTarget.call(this, entity, options[index].outcome);
  };
}

// Odometer step: drop trailing calls already at their last option, then take the next option of the last one left.
// Null once every outcome sequence has been run.
function nextPath(taken, counts) {
  let length = taken.length;
  while (length > 0 && taken[length - 1] === counts[length - 1] - 1) length -= 1;
  return length === 0 ? null : [...taken.slice(0, length - 1), taken[length - 1] + 1];
}

const clone = (value) => JSON.parse(JSON.stringify(value));

// Every outcome of `move` from `roundState`, one per outcome sequence, depth first: [{ p, state }, ...].
function expand(harness, settings, roundState, move) {
  const outcomes = [];
  let path = [];
  do {
    const state = clone(roundState);
    replay = { path, taken: [], counts: [], p: 1 };
    harness.doMove(settings, state, move, roundState.currentTurn, true);
    outcomes.push({ p: replay.p, state });
    path = nextPath(replay.taken, replay.counts);
  } while (path);
  return outcomes;
}

const snapshot = (roundState) => ({
  quantumState: roundState.quantumState,
  measuredSquares: roundState.measuredSquares,
  emptySquares: roundState.emptySquares,
  decoherenceCounters: roundState.decoherenceCounters,
  currentTurn: roundState.currentTurn,
  winner: roundState.winner,
});

function mulberry32(seed) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The outcome whose slice of [0, 1) contains u (the last one if rounding leaves a gap).
function drawOutcome(outcomes, u) {
  let rest = u;
  for (const outcome of outcomes) {
    rest -= outcome.p;
    if (rest < 0) return outcome;
  }
  return outcomes[outcomes.length - 1];
}

// Random games, X-first and O-first alternating; yields a record for every non-terminal state visited. A fresh round
// has winner null, which counts as unfinished like 0.
function* walk(harness, settings, games, rng) {
  for (let game = 0; game < games; game += 1) {
    let round = harness.createFreshRound(game % 2 ? 2 : 1);
    for (let ply = 0; ply < MAX_PLIES && !round.winner; ply += 1) {
      const legal = harness.getAvailableMoves(settings, round);
      if (legal.length === 0) throw new Error(`game ${game}, ply ${ply}: unfinished state without a legal move`);
      const children = legal.map((move) => ({ move, outcomes: expand(harness, settings, round, move) }));
      yield {
        settings: { quantumness: settings.quantumness, decoherence: settings.decoherence },
        state: snapshot(round),
        legal,
        children: children.map(({ move, outcomes }) => ({
          move,
          outcomes: outcomes.map(({ p, state }) => ({ p, state: snapshot(state) })),
        })),
      };
      const played = children[Math.floor(rng() * children.length)];
      round = drawOutcome(played.outcomes, rng()).state;
    }
  }
}

function readOptions() {
  const names = ["engine", "harness", "quantumness", "decoherence", "games", "seed"];
  const { values } = parseArgs({ options: Object.fromEntries(names.map((name) => [name, { type: "string" }])) });
  const integer = (name, min, max) => {
    const value = Number(values[name]);
    if (!Number.isInteger(value) || value < min || value > max) {
      throw new Error(`--${name} must be an integer from ${min} to ${max}`);
    }
    return value;
  };
  for (const name of ["engine", "harness"]) {
    if (!values[name]) throw new Error(`--${name} is required`);
  }
  return {
    engine: values.engine,
    harness: values.harness,
    quantumness: integer("quantumness", 0, 4),
    decoherence: integer("decoherence", 0, 3),
    games: integer("games", 1, 1e6),
    seed: integer("seed", 0, 2 ** 32 - 1),
  };
}

const options = readOptions();
const engine = await import(pathToFileURL(resolve(options.engine)).href);
const harness = loadHarness(options.harness, engine);
forceMeasurements(engine.JointQuantumSystem);
const { quantumness, decoherence } = options;
const settings = { quantumness, decoherence, timerEnabled: true, moveTime: 5 };

let records = 0;
let pairs = 0;
for (const record of walk(harness, settings, options.games, mulberry32(options.seed))) {
  process.stdout.write(`${JSON.stringify(record)}\n`);
  records += 1;
  pairs += record.legal.length;
}
process.stderr.write(`q${quantumness}-d${decoherence}: ${records} records, ${pairs} (state, move) pairs\n`);
