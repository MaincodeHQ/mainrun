import { execSync } from "node:child_process";
import fs from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

const run = (cmd) => execSync(cmd, { cwd: repoRoot, encoding: "utf-8" }).trim();

try {
  const logPath = path.join(repoRoot, "mainrun", "logs", "mainrun.log");
  if (existsSync(logPath)) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5);
    const newLogPath = path.join(repoRoot, "mainrun", "logs", `mainrun_${timestamp}.log`);
    await fs.rename(logPath, newLogPath);
    console.log(`Moved existing log to: mainrun_${timestamp}.log`);
  }

  run("git add .");

  const status = run("git status --porcelain");
  if (status === "") {
    console.log("No changes to checkpoint");
    process.exit(0);
  }

  run('git commit -m "Mainrun auto checkpoint"');
  console.log("Auto checkpoint created");
} catch (error) {
  if (error.message.includes("nothing to commit")) {
    console.log("No changes to checkpoint");
    process.exit(0);
  }

  console.error("Failed to create checkpoint:", error.message);
  process.exit(1);
}
