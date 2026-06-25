import { execSync } from "node:child_process";
import fs from "node:fs/promises";
import { existsSync } from "node:fs";
import readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");

const question = async (prompt) => {
  const rl = readline.createInterface({ input: stdin, output: stdout });
  const answer = await rl.question(prompt);
  rl.close();
  return answer;
};

const run = (cmd) => execSync(cmd, { cwd: repoRoot, encoding: "utf-8" }).trim();

try {
  let email;
  const envPath = path.join(repoRoot, ".env");

  if (existsSync(envPath)) {
    const envContent = await fs.readFile(envPath, "utf-8");
    const match = envContent.match(/EMAIL=(.+)/);
    if (match) {
      email = match[1].trim();
      console.log(`Using email: ${email}`);
    }
  }

  if (!email) {
    email = await question("Please enter your email address: ");

    await fs.writeFile(envPath, `EMAIL=${email}\n`);
    console.log("Email saved for future submissions");
  }

  console.log("\n" + "=".repeat(60));
  console.log("LEGAL NOTICE");
  console.log("=".repeat(60));
  console.log("\nBy submitting this assessment, you agree that:");
  console.log("- All submitted code becomes the property of Maincode Pty Ltd");
  console.log("- You assign all intellectual property rights to Maincode Pty Ltd");
  console.log("- You have read and agree to the full legal terms");
  console.log("\nFull terms: https://github.com/maincodehq/mainrun/blob/main/LEGAL-NOTICE.md");
  console.log("=".repeat(60) + "\n");

  const confirmation = await question("Do you agree to these terms and want to proceed? (yes/no): ");

  if (confirmation.toLowerCase() !== "yes") {
    console.log("Submission cancelled.");
    process.exit(0);
  }

  console.log("\nCreating submission zip...");

  const zipPath = path.join(repoRoot, "submission.zip");
  if (existsSync(zipPath)) await fs.unlink(zipPath);

  run(
    `zip -r submission.zip . -x "node_modules/*" -x "mainrun/data/*" -x "mainrun/.venv/*" -x ".git/*" -x "*.zip" -x "*/__pycache__/*" -x "*.DS_Store"`
  );

  console.log("Requesting upload URL...");

  const uploadUrl = run(
    `curl -s "https://api.hanger.maincode.com/api/v1/upload/request?email=${email}&filename=submission.zip"`
  );

  if (!uploadUrl || uploadUrl.includes("error")) {
    throw new Error(`Failed to get upload URL: ${uploadUrl}`);
  }

  console.log("Uploading submission...");

  run(`curl -X PUT "${uploadUrl}" --upload-file "${zipPath}"`);

  await fs.unlink(zipPath);

  console.log("✓ Submission uploaded successfully!");
} catch (error) {
  const zipPath = path.join(repoRoot, "submission.zip");
  if (existsSync(zipPath)) await fs.unlink(zipPath).catch(() => {});

  console.error("Failed to submit:", error.message);
  process.exit(1);
}
