// The JS port had `tsc` and no linter, so the checks the type system does not
// make were nobody's job: an unused import, a floating promise, a `==` where a
// `===` was meant.
//
// Type-aware rules on purpose. Without a project reference this is a syntax
// checker wearing a linter's name, and the two rules worth having here —
// `no-floating-promises` and `no-misused-promises` — need types to see anything
// at all. The engine is async and its callers await it; a dropped await is the
// bug class this exists to catch.
import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**", "data/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    rules: {
      // The port mirrors Python function for function, including names the
      // Python side chose. A linter renaming them would break the one property
      // that makes five implementations reviewable side by side.
      "@typescript-eslint/naming-convention": "off",
      // `speechText` reads JSON the fixture generator wrote; asserting its
      // shape at every access would be noise over data this repo produces.
      "@typescript-eslint/no-unsafe-assignment": "off",
      "@typescript-eslint/no-unsafe-member-access": "off",
      "@typescript-eslint/no-unsafe-argument": "off",
    },
  },
  {
    // `@huggingface/tokenizers` ships types this resolver cannot follow, so
    // every call through it reads as `any` to the linter. Turning the two rules
    // off for this one file is narrower than an `any` cast at each call site,
    // and the calls are three lines that a type would not make safer.
    // The grammar readers annotate `numbers.json` as `Record<string, any>`:
    // its value shapes differ per key by design (a scale is an object, `ones`
    // an array, `word_join` a string), and a union that described all of them
    // would be longer than the code reading it.
    files: ["src/dates.ts", "src/letters.ts", "src/numbers.ts"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unsafe-call": "off",
    },
  },
  {
    files: ["src/frontend.ts"],
    rules: {
      "@typescript-eslint/no-unsafe-call": "off",
      "@typescript-eslint/no-unsafe-return": "off",
    },
  },
  {
    // `node:test` is called and not awaited, by its own documented usage, so
    // `no-floating-promises` fires on every `test(...)` in the suite — 88 of
    // them. Off here and on in `src/`, which is where a dropped await is
    // actually a bug: the engine is async and its callers await it.
    files: ["src/**/*.test.ts", "src/test/**/*.ts"],
    rules: {
      "@typescript-eslint/no-floating-promises": "off",
      // Test helpers read the fixture JSON this repo generates and poke at
      // engine internals a public type does not describe. `any` there is the
      // honest annotation; `no-unused-vars` stays on and found a real one.
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unsafe-call": "off",
      "@typescript-eslint/no-unsafe-return": "off",
    },
  },
);
