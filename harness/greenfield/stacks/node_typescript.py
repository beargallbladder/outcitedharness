from __future__ import annotations

import json
from pathlib import Path

from harness.greenfield.models import GreenfieldManifest
from harness.greenfield.package_policy import (
    PackageAction,
    execute_package_action,
    package_name,
)
from harness.greenfield.stacks.base import StackAdapter
from harness.repo_contract import RepoContract


class NodeTypeScriptAdapter(StackAdapter):
    stack = "node-typescript"

    def bootstrap(self, root: Path, manifest: GreenfieldManifest) -> RepoContract:
        dependencies = {
            package_name(item): "*"
            for item in manifest.approved_dependencies
        }
        package = {
            "name": manifest.project_name,
            "version": "0.1.0",
            "private": True,
            "type": "module",
            "scripts": {
                "build": "tsc",
                "typecheck": "tsc --noEmit",
                "lint": "eslint .",
                "test": "npm run build --silent && node --test dist/tests/*.test.js",
            },
            "dependencies": dependencies,
            "devDependencies": {
                "@eslint/js": "*",
                "@types/node": "*",
                "eslint": "*",
                "typescript": "*",
                "typescript-eslint": "*",
            },
        }
        self.write_file(root, "package.json", json.dumps(package, indent=2) + "\n")
        self.write_file(
            root,
            "tsconfig.json",
            json.dumps(
                {
                    "compilerOptions": {
                        "target": "ES2022",
                        "module": "NodeNext",
                        "moduleResolution": "NodeNext",
                        "rootDir": ".",
                        "outDir": "dist",
                        "strict": True,
                        "noUncheckedIndexedAccess": True,
                        "esModuleInterop": True,
                        "skipLibCheck": True,
                        "types": ["node"],
                    },
                    "include": ["src/**/*.ts", "tests/**/*.ts"],
                },
                indent=2,
            )
            + "\n",
        )
        self.write_file(
            root,
            "eslint.config.js",
            """import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  js.configs.recommended,
  ...tseslint.configs.recommended,
  { ignores: ["dist/**"] },
);
""",
        )
        self.write_file(
            root,
            "src/index.ts",
            """export function health() {
  return { status: "ok" };
}
""",
        )
        self.write_file(
            root,
            "tests/smoke.test.ts",
            """import assert from "node:assert/strict";
import test from "node:test";

import { health } from "../src/index.js";

test("health smoke", () => {
  assert.deepEqual(health(), { status: "ok" });
});
""",
        )
        self.write_file(
            root,
            ".harness.toml",
            """[verification]
required = ["test", "lint", "typecheck"]

[verification.commands.test]
argv = ["npm", "run", "test"]
timeout = 120

[verification.commands.lint]
argv = ["npm", "run", "lint"]
timeout = 120

[verification.commands.typecheck]
argv = ["npm", "run", "typecheck"]
timeout = 120
""",
        )
        self.write_file(
            root,
            ".gitignore",
            "node_modules/\ndist/\ncoverage/\n.env\n*.log\n",
        )
        self.write_file(
            root,
            ".env.example",
            "# Add documented non-secret configuration only when required.\n",
        )
        self.write_file(
            root,
            "README.md",
            f"# {manifest.project_name}\n\n"
            "Autonomously bootstrapped and mechanically verified by Harness.\n",
        )
        execute_package_action(
            PackageAction("npm", ("npm", "install", "--ignore-scripts"), root),
            repo_root=root,
            approved_dependencies=manifest.approved_dependencies,
        )
        contract = self.contract(root)
        self.verify(contract)
        return contract
