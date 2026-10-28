"""A scripted stand-in for Claude, so the real CrewAI agents and tools run in tests without any API call.

Each agent (by role) follows its own script. A step is one of:
  tool("name", {...})  -> the agent asks CrewAI to run that tool (ReAct text format)
  final("text")        -> the agent's final answer
  a callable(messages) -> returns one of the above at call time (e.g. built from a file the tools wrote)
If a script runs out, the agent repeats its last step.
"""
import json

from crewai import BaseLLM


def tool(name: str, args: dict | None = None) -> str:
    return f"Thought: I will use the {name} tool.\nAction: {name}\nAction Input: {json.dumps(args or {})}"


def final(text: str) -> str:
    return f"Thought: I have what I need.\nFinal Answer: {text}"


class ScriptedLLM(BaseLLM):
    scripts: dict = {}
    position: dict = {}
    calls: list = []
    fail_with: object = None  # an exception to raise on every call (API failure simulation)

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None,
             from_agent=None, response_model=None):
        role = from_agent.role if from_agent is not None else "unknown"
        self.calls.append({"role": role, "messages": messages})
        if self.fail_with is not None:
            raise self.fail_with
        script = self.scripts[role]
        index = self.position.get(role, 0)
        self.position[role] = index + 1
        step = script[min(index, len(script) - 1)]
        text = step(messages) if callable(step) else step
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages) if isinstance(messages, list) \
            else len(str(messages))
        self._track_token_usage_internal({"prompt_tokens": prompt_chars // 4, "completion_tokens": len(text) // 4,
                                          "total_tokens": prompt_chars // 4 + len(text) // 4})
        return text

    def supports_function_calling(self) -> bool:
        return False  # use the ReAct text protocol, which a script can speak


def scripted_llm(scripts: dict, model: str = "mock/scripted", fail_with=None) -> ScriptedLLM:
    return ScriptedLLM(model=model, scripts=scripts, position={}, calls=[], fail_with=fail_with)
