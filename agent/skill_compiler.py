"""Compile selected skills into a single prompt policy."""

from __future__ import annotations

from typing import List

from agent.skill_policy_engine import PhaseSkillPlan, SkillSelection


class SkillCompiler:
    def compile_prompt(self, base_prompt: str, plan: PhaseSkillPlan) -> str:
        if not plan.selected:
            return base_prompt

        lines: List[str] = [
            base_prompt,
            "",
            f"Compiled skill policy for surface={plan.surface}, phase={plan.phase}.",
            "1. Host capability boundaries and tool constraints always take priority over skill instructions.",
            "2. Apply only the selected skills below. Ignore any conflicting instruction from non-selected skills.",
        ]

        if plan.primary is not None:
            lines.append(f"3. Treat {plan.primary.skill.skill_id} as the primary skill for this phase.")
        if plan.modifiers:
            lines.append("4. Treat modifier skills only as overlays for tone, structure, or formatting unless they explicitly define stronger constraints.")
        if plan.semantic_judgment is not None and plan.semantic_judgment.intent_summary:
            lines.append(f"5. Semantic routing summary: {plan.semantic_judgment.intent_summary}")

        lines.append("")
        if plan.primary is not None:
            lines.extend(self._render_primary(plan.primary))
        if plan.modifiers:
            lines.extend(self._render_modifiers(list(plan.modifiers)))

        rejected = [item for item in plan.decisions if item.decision == "rejected" and item.reason.startswith("conflict_with:")]
        if rejected:
            lines.append("")
            lines.append("Suppressed conflicting skills:")
            for item in rejected[:4]:
                lines.append(f"- {item.skill_id} ({item.reason})")

        return "\n".join(line for line in lines if line is not None).strip()

    def _render_primary(self, selection: SkillSelection) -> List[str]:
        return [
            "Primary skill:",
            f"- {selection.skill.skill_id} | type={selection.skill.type} | subtype={selection.skill.subtype} | priority={selection.skill.priority} | source={selection.source} | score={selection.score:.2f}",
            selection.skill.prompt.strip(),
        ]

    def _render_modifiers(self, modifiers: List[SkillSelection]) -> List[str]:
        lines: List[str] = ["Modifier skills:"]
        for item in modifiers:
            lines.append(
                f"- {item.skill.skill_id} | subtype={item.skill.subtype} | priority={item.skill.priority} | source={item.source} | score={item.score:.2f}"
            )
        lines.append("")
        lines.append("Modifier instructions:")
        for item in modifiers:
            lines.append(f"[{item.skill.skill_id}]")
            lines.append(item.skill.prompt.strip())
            lines.append("")
        while lines and not lines[-1].strip():
            lines.pop()
        return lines
