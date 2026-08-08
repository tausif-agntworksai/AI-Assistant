"""The skill registry.

One decorator is the single source of truth for three consumers that would
otherwise drift apart:

  1. the offline rule matcher (via `examples` and `name`)
  2. Claude's tool schema (via type hints and `params`)
  3. the permission gate (via `risk`)

Adding a capability means writing one function. Nothing else needs updating.
"""

from __future__ import annotations

import enum
import inspect
import logging
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_args, get_origin

from ..permissions import PermissionDenied, Risk, audit, gate

log = logging.getLogger(__name__)


# --- results ---------------------------------------------------------------


@dataclass
class Reply:
    """A spoken response in both languages.

    Skills provide both so the assistant can answer in whichever language it
    was addressed in without a translation round-trip.
    """

    en: str
    hi: str = ""

    def text(self, language: str = "en") -> str:
        if (language or "").lower().startswith("hi") and self.hi:
            return self.hi
        return self.en

    def __str__(self) -> str:
        return self.en


@dataclass
class SkillResult:
    ok: bool
    reply: Reply
    data: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def text(self, language: str = "en") -> str:
        return self.reply.text(language)


def ok(en: str, hi: str = "", detail: str = "", **data: Any) -> SkillResult:
    return SkillResult(True, Reply(en, hi), data, detail)


def fail(en: str, hi: str = "", detail: str = "", **data: Any) -> SkillResult:
    return SkillResult(False, Reply(en, hi), data, detail)


@dataclass
class SkillContext:
    """Ambient information a skill may need. Injected, never part of the schema."""

    language: str = "en"
    transcript: str = ""
    source: str = "voice"  # voice | text | hud
    dry_run: bool = False


# --- specification ---------------------------------------------------------


@dataclass
class SkillSpec:
    name: str
    func: Callable[..., SkillResult]
    description: str
    risk: Risk = Risk.SAFE
    params: dict[str, str] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    category: str = "general"
    confirm_en: str = ""
    confirm_hi: str = ""
    hidden: bool = False  # kept out of Claude's tool list

    def __post_init__(self) -> None:
        self.signature = inspect.signature(self.func)
        try:
            self.hints = typing.get_type_hints(self.func)
        except Exception:  # noqa: BLE001 - forward refs we can't resolve
            self.hints = {}
        self.wants_context = any(
            p.annotation is SkillContext or name == "ctx"
            for name, p in self.signature.parameters.items()
        )

    def confirm_prompts(self, args: dict[str, Any]) -> tuple[str, str]:
        """Build the confirmation question, interpolating the arguments."""
        default_en = f"{self.description}. Confirm?"
        default_hi = f"{self.description}. Confirm karein?"
        try:
            en = (self.confirm_en or default_en).format(**args)
            hi = (self.confirm_hi or default_hi).format(**args)
        except (KeyError, IndexError):
            en, hi = default_en, default_hi
        return en, hi

    def json_schema(self) -> dict[str, Any]:
        """Anthropic tool schema, derived from the function signature."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for pname, param in self.signature.parameters.items():
            if pname in ("ctx", "self") or param.annotation is SkillContext:
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            schema = _type_to_schema(self.hints.get(pname, str))
            if pname in self.params:
                schema["description"] = self.params[pname]
            if param.default is not inspect.Parameter.empty and param.default is not None:
                schema["default"] = param.default
            properties[pname] = schema

            if param.default is inspect.Parameter.empty:
                required.append(pname)

        description = self.description
        if self.examples:
            description += "\nExample phrasings: " + "; ".join(f'"{e}"' for e in self.examples[:6])

        return {
            "name": self.name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    """Map a Python type hint onto a JSON Schema fragment."""
    origin = get_origin(annotation)

    if origin is Literal:
        values = list(get_args(annotation))
        kind = "number" if values and isinstance(values[0], (int, float)) else "string"
        return {"type": kind, "enum": values}

    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        # Optional[X] -> schema for X; the gate treats missing as absent anyway.
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return _type_to_schema(non_none[0]) if non_none else {"type": "string"}

    if origin in (list, tuple):
        args = get_args(annotation)
        return {"type": "array", "items": _type_to_schema(args[0]) if args else {"type": "string"}}

    if origin is dict:
        return {"type": "object"}

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {"type": "string", "enum": [m.value for m in annotation]}

    return {
        bool: {"type": "boolean"},
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
    }.get(annotation, {"type": "string"})


# --- registry --------------------------------------------------------------


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def add(self, spec: SkillSpec) -> None:
        if spec.name in self._skills:
            log.warning("Skill %r redefined", spec.name)
        self._skills[spec.name] = spec

    def get(self, name: str) -> SkillSpec | None:
        return self._skills.get(name)

    def all(self) -> list[SkillSpec]:
        return list(self._skills.values())

    def by_category(self) -> dict[str, list[SkillSpec]]:
        out: dict[str, list[SkillSpec]] = {}
        for spec in self._skills.values():
            out.setdefault(spec.category, []).append(spec)
        return out

    def names(self) -> list[str]:
        return sorted(self._skills)

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [s.json_schema() for s in self._skills.values() if not s.hidden]

    def execute(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        ctx: SkillContext | None = None,
    ) -> SkillResult:
        """Run a skill. The one place permissions are enforced and actions logged."""
        args = dict(args or {})
        ctx = ctx or SkillContext()
        spec = self._skills.get(name)

        if spec is None:
            audit(name, allowed=False, ok=False, detail="unknown skill",
                  source=ctx.source, transcript=ctx.transcript)
            return fail(f"I don't know how to {name.replace('_', ' ')}.",
                        f"Mujhe {name.replace('_', ' ')} karna nahi aata.")

        args = self._coerce_args(spec, args)

        if ctx.dry_run:
            detail = f"{spec.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
            log.info("DRY RUN: %s [risk=%s]", detail, spec.risk.value)
            audit(spec.name, risk=spec.risk.value, args=args, allowed=True, ok=True,
                  detail="dry-run: " + detail, source=ctx.source, transcript=ctx.transcript)
            return ok(f"Dry run: would {detail}", detail=detail)

        prompt_en, prompt_hi = spec.confirm_prompts(args)
        if not gate.check(spec.name, spec.risk, prompt_en, prompt_hi, ctx.language):
            audit(spec.name, risk=spec.risk.value, args=args, allowed=False, ok=False,
                  detail="denied at confirmation", source=ctx.source, transcript=ctx.transcript)
            return fail("Cancelled.", "Theek hai, rehne dete hain.")

        if spec.wants_context:
            args["ctx"] = ctx

        try:
            result = spec.func(**args)
        except PermissionDenied as exc:
            audit(spec.name, risk=spec.risk.value, args=args, allowed=False, ok=False,
                  detail=str(exc), source=ctx.source, transcript=ctx.transcript)
            return fail("I'm not allowed to do that.", "Mujhe iski ijazat nahi hai.")
        except TypeError as exc:
            log.error("Bad arguments for %s: %s", spec.name, exc)
            audit(spec.name, risk=spec.risk.value, args=args, allowed=True, ok=False,
                  detail=f"TypeError: {exc}", source=ctx.source, transcript=ctx.transcript)
            return fail("I didn't catch what to apply that to.",
                        "Samajh nahi aaya kis cheez pe karna hai.")
        except Exception as exc:  # noqa: BLE001 - one bad skill must not kill the engine
            log.exception("Skill %s failed", spec.name)
            audit(spec.name, risk=spec.risk.value, args=args, allowed=True, ok=False,
                  detail=f"{type(exc).__name__}: {exc}", source=ctx.source,
                  transcript=ctx.transcript)
            return fail("That didn't work.", "Ye kaam nahi ho paya.")

        if not isinstance(result, SkillResult):
            result = ok(str(result))

        audit(spec.name, risk=spec.risk.value, args={k: v for k, v in args.items() if k != "ctx"},
              allowed=True, ok=result.ok, detail=result.detail or result.reply.en,
              source=ctx.source, transcript=ctx.transcript)
        return result

    @staticmethod
    def _coerce_args(spec: SkillSpec, args: dict[str, Any]) -> dict[str, Any]:
        """Coerce values to their annotated types.

        Speech recognition and LLMs both hand back strings for everything, so
        `set_volume(level="50")` needs to reach an `int` parameter as 50.
        """
        out: dict[str, Any] = {}
        for key, value in args.items():
            if key not in spec.signature.parameters:
                continue  # silently drop hallucinated arguments
            target = spec.hints.get(key)
            try:
                if target is int and not isinstance(value, bool):
                    out[key] = int(float(value))
                elif target is float:
                    out[key] = float(value)
                elif target is bool and isinstance(value, str):
                    out[key] = value.strip().lower() in ("true", "yes", "1", "on", "haan", "ha")
                else:
                    out[key] = value
            except (TypeError, ValueError):
                out[key] = value
        return out


registry = SkillRegistry()


def skill(
    name: str,
    description: str,
    risk: Risk = Risk.SAFE,
    params: dict[str, str] | None = None,
    examples: list[str] | None = None,
    category: str = "general",
    confirm_en: str = "",
    confirm_hi: str = "",
    hidden: bool = False,
) -> Callable[[Callable[..., SkillResult]], Callable[..., SkillResult]]:
    """Register a function as a skill. See the module docstring."""

    def decorator(func: Callable[..., SkillResult]) -> Callable[..., SkillResult]:
        registry.add(
            SkillSpec(
                name=name,
                func=func,
                description=description,
                risk=risk,
                params=params or {},
                examples=examples or [],
                category=category,
                confirm_en=confirm_en,
                confirm_hi=confirm_hi,
                hidden=hidden,
            )
        )
        return func

    return decorator
