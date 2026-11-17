#!/usr/bin/env node
// Plays random games with the organisers' Cheqqers harness and prints one JSON line per move, so that the Python
// simulator can be checked against the harness itself (tests/compliance/test_cheqqers_harness_differential.py).
//
//   node cheqqers_trace.mjs <path to local-cheqqers-server.js> <games> <seed> [max plies] [split bias] [limit]
//
// [split bias] (0..1, default 0): the chance of choosing among the split moves when there are any. Random play
// rarely entangles more than a few squares; a high bias reaches the engine's limit on entangled squares, where
// the harness swallows the engine's error and the turn does not pass.
// [limit]: the engine's limit on the dimension of an entangled system (default: the engine's own, 32768 = 15
// squares). A low limit, set to the same number in the simulator, makes that rare case common enough to test.
//
// The harness file is a command-line script: its classes are not exported and its last lines start a game. This
// driver copies the file next to the original (so that the engine in that directory's node_modules resolves), cuts
// the start-up lines, appends an export, and imports the copy. Math.random is replaced by a seeded generator before
// anything is imported, so a (harness, games, seed) triple always gives the same games.
//
// Each line: { game, ply, multiJump, payload, moveIndex, move, engineError, after }.
//   payload   what fetchRemoteAiMove would POST: board, quantumStates, legalMoves, player, settings
//   multiJump the square a capture sequence continues from, or null (the payload does not carry it)
//   move      the chosen legal move as the harness lists it; after: the position once it is applied
import fs from 'fs';
import path from 'path';
import { pathToFileURL } from 'url';

const [harnessPath, gamesArg, seedArg, maxPliesArg, splitBiasArg, limitArg] = process.argv.slice(2);
if (!harnessPath || !gamesArg || !seedArg) {
    console.error('usage: node cheqqers_trace.mjs <local-cheqqers-server.js> <games> <seed> [max plies]');
    process.exit(2);
}
const games = parseInt(gamesArg, 10);
const maxPlies = parseInt(maxPliesArg || '200', 10);
const splitBias = parseFloat(splitBiasArg || '0');

// mulberry32
let s = parseInt(seedArg, 10) >>> 0;
Math.random = function () {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const source = fs.readFileSync(harnessPath, 'utf8');
const startup = source.lastIndexOf('const config = parseArgs();');
if (startup < 0) {
    console.error('the harness does not end with the expected start-up lines');
    process.exit(2);
}
const library = source.slice(0, startup) +
    '\nexport { CheqqersGame, initEngine, moveToArray, serializeBoardState, PieceColor, GameState, GameType,' +
    ' JointQuantumSystem };\n';
const copy = path.join(path.dirname(path.resolve(harnessPath)), `.cheqqers_trace_${process.pid}.mjs`);
fs.writeFileSync(copy, library);
let harness;
try {
    harness = await import(pathToFileURL(copy).href);
} finally {
    fs.unlinkSync(copy);
}
const { CheqqersGame, initEngine, moveToArray, serializeBoardState, PieceColor, GameState, GameType } = harness;
await initEngine();
if (limitArg) harness.JointQuantumSystem.setMaxDimensionLimit(parseInt(limitArg, 10));
// The harness reports a swallowed engine error on stderr; count them so that a run can say it met the case.
let swallowed = 0;
const realError = console.error;
console.error = (...args) => { if (String(args[0]).includes('Move application error')) swallowed++; else realError(...args); };

function position(game) {
    const state = game.toGameStateObject();
    return {
        turn: game.turn === PieceColor.WHITE ? 'white' : 'black',
        state: game.state,
        board: serializeBoardState(state),
        quantumStates: state.quantum_states,
        legal: state.possible_moves,
    };
}

for (let g = 0; g < games; g++) {
    const game = new CheqqersGame(6, 2, GameType.ENTANGLEMENT);
    for (let ply = 0; ply < maxPlies && game.state === GameState.IN_PROGRESS; ply++) {
        const before = position(game);
        if (before.legal.length === 0) break;
        let moveIndex = Math.floor(Math.random() * before.legal.length);
        if (splitBias > 0) {
            const splits = before.legal.map((m, i) => ('to_index1' in m ? i : -1)).filter(i => i >= 0);
            if (splits.length > 0 && Math.random() < splitBias) moveIndex = splits[Math.floor(Math.random() * splits.length)];
        }
        const record = {
            game: g,
            ply,
            multiJump: game.multiJumpSquare,
            payload: {
                player: before.turn,
                currentTurn: before.turn,
                board: before.board,
                quantumStates: before.quantumStates,
                settings: { boardSize: game.boardSize, quantumness: game.gameType },
                legalMoves: before.legal.map(moveToArray),
            },
            moveIndex,
            move: moveToArray(before.legal[moveIndex]),
        };
        const errorsBefore = swallowed;
        game.applyMove(moveIndex);
        record.engineError = swallowed > errorsBefore;  // the engine refused the move and the harness swallowed it
        const after = position(game);
        record.after = { turn: after.turn, state: after.state, board: after.board, multiJump: game.multiJumpSquare };
        process.stdout.write(JSON.stringify(record) + '\n');
    }
}
realError(`swallowed engine errors: ${swallowed}`);
