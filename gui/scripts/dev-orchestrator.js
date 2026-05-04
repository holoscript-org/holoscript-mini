#!/usr/bin/env node
"use strict";

/**
 * HoloScript dev orchestrator.
 * Starts backend and Next.js GUI concurrently.
 * All processes share stdout/stderr with clear prefixes and terminate together.
 */

const { spawn } = require("child_process");
const path      = require("path");
const os        = require("os");

// ── Path resolution ───────────────────────────────────────────────────────────
const GUI_DIR  = path.resolve(__dirname, "..");
const ROOT_DIR = path.resolve(GUI_DIR, "..");

// ── Platform detection ────────────────────────────────────────────────────────
const IS_WIN = os.platform() === "win32";
const PY     = IS_WIN ? "python"   : "python3";
const NPM    = IS_WIN ? "npm.cmd"  : "npm";

// ── ANSI colour codes ─────────────────────────────────────────────────────────
const CLR = {
  reset:    "\x1b[0m",
  backend:  "\x1b[36m",   // cyan
  renderer: "\x1b[35m",   // magenta
  gui:      "\x1b[32m",   // green
  orch:     "\x1b[33m",   // yellow
  err:      "\x1b[31m",   // red
};

function tag(label, color) {
  return `${color}[${label}]${CLR.reset}`;
}

// ── Process spawner ───────────────────────────────────────────────────────────
function spawnProc(label, color, cmd, args, cwd) {
  const proc = spawn(cmd, args, {
    cwd,
    // shell required on Windows for .cmd wrappers; on Unix keep false for
    // clean signal delivery and proper process-group kill.
    shell:    IS_WIN,
    // detached on Unix: creates its own process group so we can kill -PID
    detached: !IS_WIN,
    stdio:    ["ignore", "pipe", "pipe"],
    env:      { ...process.env, PYTHONUNBUFFERED: "1", FORCE_COLOR: "1" },
  });

  const forward = (stream, write) => {
    stream.on("data", (chunk) => {
      chunk
        .toString()
        .split(/\r?\n/)
        .filter(Boolean)
        .forEach((line) => write(`${tag(label, color)} ${line}\n`));
    });
  };

  forward(proc.stdout, (s) => process.stdout.write(s));
  forward(proc.stderr, (s) => process.stderr.write(s));

  proc.on("error", (err) => {
    process.stderr.write(
      `${tag(label, CLR.err)} spawn error: ${err.message}\n`
    );
  });

  return proc;
}

process.on("uncaughtException", (err) => {
  process.stderr.write(`${tag("ORCH", CLR.err)} uncaughtException: ${err.stack || err.message}\n`);
});

process.on("unhandledRejection", (reason) => {
  const message = reason instanceof Error ? reason.stack || reason.message : String(reason);
  process.stderr.write(`${tag("ORCH", CLR.err)} unhandledRejection: ${message}\n`);
});

// ── Launch services ───────────────────────────────────────────────────────────
process.stdout.write(
  `${tag("ORCH", CLR.orch)} HoloScript starting — root: ${ROOT_DIR}\n`
);
process.stdout.write(
  `${tag("ORCH", CLR.orch)} Press Ctrl+C to stop all services\n\n`
);

const backend = spawnProc(
  "BACKEND", CLR.backend,
  PY,
  [
    "-m", "uvicorn",
    "backend.api_server:app",
    "--port", "8000",
    "--log-level", "warning",
  ],
  ROOT_DIR
);

const gui = spawnProc(
  "GUI", CLR.gui,
  NPM,
  ["run", "next-dev"],
  GUI_DIR
);

const ALL = [
  { proc: backend,  label: "BACKEND"  },
  { proc: gui,      label: "GUI"      },
];

// ── Shutdown coordinator ──────────────────────────────────────────────────────
let exiting = false;

function killAll(reason) {
  if (exiting) return;
  exiting = true;

  process.stdout.write(
    `\n${tag("ORCH", CLR.orch)} Shutting down (${reason})...\n`
  );

  for (const { proc, label } of ALL) {
    if (proc.exitCode !== null || proc.killed) continue;
    try {
      if (IS_WIN) {
        // /T kills the full child-process tree (cmd.exe + python / node)
        spawn("taskkill", ["/F", "/T", "/PID", String(proc.pid)], {
          shell: true,
          stdio: "ignore",
        });
      } else {
        // Negative PID kills the entire process group
        process.kill(-proc.pid, "SIGTERM");
      }
    } catch (err) {
      // Process may have already exited — ignore
      process.stderr.write(
        `${tag(label, CLR.err)} kill failed: ${err.message}\n`
      );
    }
  }

  // Give processes a moment to clean up, then force exit
  setTimeout(() => process.exit(0), 2000);
}

// ── Crash propagation ─────────────────────────────────────────────────────────
for (const { proc, label } of ALL) {
  proc.on("exit", (code, signal) => {
    if (exiting) return;
    process.stdout.write(
      `${tag(label, CLR.err)} exited` +
      ` (code=${code ?? "—"} signal=${signal ?? "—"})\n`
    );
    killAll(`${label} exited unexpectedly`);
  });
}

// ── Signal handlers ───────────────────────────────────────────────────────────
process.on("SIGINT",  () => killAll("SIGINT"));
process.on("SIGTERM", () => killAll("SIGTERM"));
