"""Policy-as-code: declarative rules deciding whether a tool call is allowed, denied or needs approval.

A policy is data (YAML or a dict), validated at load time, so it can be reviewed, versioned and tested
like code. Rules are evaluated in order and the first match wins. When no rule matches, a built-in
safety baseline applies: write or destructive tools of *external* toolsets require human approval.
An explicit rule can override the baseline, which makes relaxing it a visible, reviewable decision.

    version: 1
    default: allow
    rules:
      - id: prod-is-read-only
        tools: ["gitlab_*"]
        effects: [write, destructive]
        when:
          - {arg: project, op: matches, value: "^prod/"}
        decision: deny
        reason: Production projects are read-only for agents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .tools import Effect, Tool, Toolset


class PolicyError(ValueError):
    """The policy file is malformed or inconsistent."""


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    rule_id: str | None
    reason: str


_MISSING = object()


def _lookup(args: dict, path: str) -> Any:
    value: Any = args
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


class Condition(BaseModel):
    """A test on one tool argument. A missing argument never satisfies a condition."""
    model_config = ConfigDict(extra="forbid")

    arg: str = Field(min_length=1, description="Argument name; dotted path for nested values.")
    op: Literal["eq", "ne", "in", "not_in", "matches"]
    value: Any

    @model_validator(mode="after")
    def _check_value(self) -> Condition:
        if self.op in {"in", "not_in"} and not isinstance(self.value, list):
            raise ValueError(f"'{self.op}' needs a list value.")
        if self.op == "matches":
            if not isinstance(self.value, str):
                raise ValueError("'matches' needs a regular expression string.")
            try:
                re.compile(self.value)
            except re.error as exc:
                raise ValueError(f"Invalid regular expression: {exc}") from exc
        return self

    def holds(self, args: dict) -> bool:
        have = _lookup(args, self.arg)
        if have is _MISSING:
            return False
        match self.op:
            case "eq":
                return bool(have == self.value)
            case "ne":
                return bool(have != self.value)
            case "in":
                return have in self.value
            case "not_in":
                return have not in self.value
            case _:
                return re.search(self.value, str(have)) is not None


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    tools: list[str] = Field(default_factory=lambda: ["*"], description="Glob patterns on tool names.")
    toolsets: list[str] = Field(default_factory=lambda: ["*"], description="Glob patterns on toolset ids.")
    effects: list[Effect] | None = Field(default=None, description="Any effect when omitted.")
    when: list[Condition] = Field(default_factory=list, description="All conditions must hold.")
    decision: Decision
    reason: str = ""

    def applies(self, tool: Tool, toolset: Toolset | None, args: dict) -> bool:
        if not any(fnmatchcase(tool.name, pattern) for pattern in self.tools):
            return False
        if not any(fnmatchcase(toolset.id if toolset else "", pattern) for pattern in self.toolsets):
            return False
        if self.effects is not None and tool.effect not in self.effects:
            return False
        return all(condition.holds(args) for condition in self.when)


BASELINE_RULE_ID = "baseline:external-write"


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    default: Decision = Decision.ALLOW
    rules: list[Rule] = Field(default_factory=list)

    @field_validator("rules")
    @classmethod
    def _unique_ids(cls, rules: list[Rule]) -> list[Rule]:
        ids = [rule.id for rule in rules]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"Duplicate rule ids: {', '.join(duplicates)}")
        return rules

    def evaluate(self, tool: Tool, toolset: Toolset | None, args: dict) -> PolicyDecision:
        for rule in self.rules:
            if rule.applies(tool, toolset, args):
                return PolicyDecision(rule.decision, rule.id, rule.reason or f"Matched rule {rule.id}.")
        if toolset is not None and toolset.external and tool.effect != Effect.READ:
            return PolicyDecision(Decision.REQUIRE_APPROVAL, BASELINE_RULE_ID,
                                  "Changes to external systems need human approval.")
        return PolicyDecision(self.default, None, "No rule matched; policy default applies.")

    @classmethod
    def from_dict(cls, data: dict) -> Policy:
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise PolicyError(_describe(exc)) from exc

    @classmethod
    def from_yaml(cls, source: str | Path) -> Policy:
        """Load from a file path or a YAML string."""
        text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PolicyError(f"Invalid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise PolicyError("A policy must be a mapping.")
        return cls.from_dict(data)


def _describe(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}" for error in exc.errors())
