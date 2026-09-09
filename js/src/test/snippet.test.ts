/**
 * The ```typescript blocks on the front page and the landing page type-check
 * against the built package.
 *
 * Each block becomes a module in a scratch directory whose `node_modules/loudkit`
 * is this package, and `tsc --noEmit` reads it the way a reader's editor
 * would: through `package.json`'s `exports`, to `dist/index.d.ts`. Weight-free.
 */

import assert from "node:assert";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const PAGES = ["README.md", "site/src/handwritten/index.mdx"];

function packageRoot(): string {
  let dir = dirname(fileURLToPath(import.meta.url));
  while (!existsSync(join(dir, "package.json"))) {
    const up = dirname(dir);
    if (up === dir) throw new Error("no package.json above " + import.meta.url);
    dir = up;
  }
  return dir;
}

/** Every ```typescript or ```ts fence on a page, with a tab component's indentation removed. */
function typescriptBlocks(text: string): string[] {
  const blocks: string[] = [];
  let lines: string[] | null = null;
  for (const line of text.split("\n")) {
    const stripped = line.trimStart();
    if (lines === null) {
      if (/^```(typescript|ts)\b/.test(stripped)) lines = [];
    } else if (stripped.startsWith("```")) {
      blocks.push(dedent(lines));
      lines = null;
    } else {
      lines.push(line);
    }
  }
  return blocks;
}

function dedent(lines: string[]): string {
  const indent = Math.min(
    ...lines.filter((l) => l.trim() !== "").map((l) => l.length - l.trimStart().length)
  );
  return lines.map((l) => l.slice(Math.min(indent, l.length))).join("\n") + "\n";
}

test("the user-page typescript snippets type-check against dist", () => {
  const pkg = packageRoot();
  const repo = dirname(pkg);
  assert.ok(existsSync(join(pkg, "dist", "index.d.ts")), "dist is not built");

  const scratch = mkdtempSync(join(tmpdir(), "loudkit-snippet-"));
  try {
    mkdirSync(join(scratch, "node_modules"));
    symlinkSync(pkg, join(scratch, "node_modules", "loudkit"), "dir");
    writeFileSync(join(scratch, "package.json"), JSON.stringify({ type: "module" }));
    const files: string[] = [];
    const bodies: string[] = [];
    for (const page of PAGES) {
      const text = readFileSync(join(repo, page), "utf8");
      const blocks = typescriptBlocks(text);
      assert.ok(blocks.length > 0, `${page} shows no typescript block; the gate is looking at nothing`);
      blocks.forEach((body, i) => {
        const name = `${page.replace(/[/.]/g, "_")}_${i + 1}.ts`;
        writeFileSync(join(scratch, name), body);
        files.push(name);
        bodies.push(`${page} block ${i + 1}:\n${body}`);
      });
    }
    // Compile the actual cloning recipe as well as the landing-page hellos.
    // The engine is the one already loaded by the guide; saveVoice must be imported.
    const guide = readFileSync(join(repo, "docs/guides/07-js-ts.md"), "utf8");
    const cloning = guide.split("## Cloning a voice")[1].split("```javascript\n")[1].split("```")[0];
    const cloneFile = "guide_cloning.ts";
    writeFileSync(join(scratch, cloneFile), 'import { Engine } from "loudkit";\ndeclare const engine: Engine;\n' + cloning);
    files.push(cloneFile);
    bodies.push(`cloning guide:\n${cloning}`);
    writeFileSync(
      join(scratch, "tsconfig.json"),
      JSON.stringify({
        compilerOptions: {
          target: "ES2022",
          module: "NodeNext",
          moduleResolution: "NodeNext",
          strict: true,
          noEmit: true,
          skipLibCheck: true,
          types: ["node"],
          typeRoots: [join(pkg, "node_modules", "@types")],
        },
        files,
      })
    );
    const tsc = join(pkg, "node_modules", "typescript", "bin", "tsc");
    const run = spawnSync(process.execPath, [tsc, "-p", scratch], { encoding: "utf8" });
    assert.strictEqual(
      run.status,
      0,
      `a typescript block on a user page does not type-check:\n${run.stdout}${run.stderr}\n--- the blocks ---\n${bodies.join("\n")}`
    );
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
});
