"""Reusable system-prompt builder for SumBench MCP agents.

Provides ``build_system_prompt()`` and ``build_user_prompt()`` functions
that assemble complete agent prompts from skill-specific parameters.
Universal sections (code-execution instructions, h2ogpte-sdk bootstrap,
sub-agent patterns, no-internet warning) are baked in; everything else
is injected via keyword arguments.

Follows the same pattern as the shared prompt_template.py but defaults
to ``claude_tool_runner.py`` (required for MCP tool access).

Usage::

    from prompt_template import build_system_prompt, build_user_prompt

    SYSTEM_PROMPT = build_system_prompt(
        role="You are an expert summary evaluation agent. ...",
        skill_zip_name="sumbench_mcp.zip",
        skill_zip_contents="server.py, evaluators/, envs.json, requirements.txt, nltk_data/",
        bootstrap_bash='''pip install --quiet --no-index --find-links deps/ --no-deps -r requirements.txt
export NLTK_DATA=$(pwd)/nltk_data''',
        workspace_files=[("server.py", "MCP server entrypoint"), ...],
        domain_guide='''## MCP Evaluation Tools ...''',
        output_filename="evaluation_report.md",
        output_instructions="Verify the report contains a score table",
        key_rules=["Always pass FULL text to evaluate_summary", ...],
        include_subagents=False,
    )
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Universal sections (shared across ALL agent types)
# ---------------------------------------------------------------------------

_CODE_EXECUTION = """\
## CRITICAL: How to Execute Code

**Always write executable code in fenced markdown blocks.** The execution engine reads these blocks and runs them for you. Do NOT use function-call syntax or any other format.

Correct -- shell commands:
```bash
ls -la
pip install markitdown
```

Correct -- Python:
```python
from pathlib import Path
print(Path(".").absolute())
```

Wrong -- do not use this format (will NOT be executed):
<|tool_calls_section_begin|>...<|tool_calls_section_end|>
functions.execute_code(...)

Write all code as fenced markdown blocks. The engine will execute them and show you the output."""

_NO_INTERNET = """\
## CRITICAL: Dependency Bootstrap (No Network Access)

The sandbox has NO internet access. Install only from local wheels bundled in your skill zip:

```bash
unzip -o {skill_zip_name}
{sdk_unzip_line}pip install --quiet --no-deps {deps_path}*.whl
```

Do NOT attempt to pip install from the internet -- it will hang for 300 seconds then timeout.
Do NOT use shell `export` -- it is blocked. Use Python `os.environ` instead."""

_SUBAGENT_SECTION = """\
## Sub-Agents via h2ogpte SDK

For large tasks, spawn sub-agents to work on sections in parallel. Each sub-agent gets:
- Its own h2ogpte session in the SAME collection (all workspace files available)
- A focused task
- The connection values needed to connect

**Read `h2ogpte-sdk/references/subagent.md` for the full pattern and fan-out example.**

Quick reference:

```python
import sys; sys.path.insert(0, 'h2ogpte-sdk')
from h2ogpte import H2OGPTE
from scripts.subagent import run_subagent
from concurrent.futures import ThreadPoolExecutor, as_completed

# Use the literal values injected into your prompt below:
client = H2OGPTE(address=H2OGPTE_ADDRESS, api_key=H2OGPTE_API_KEY, verify=True)

# Fan out: process sections in parallel
section_assignments = [("1-4", "Section A"), ("5-8", "Section B")]
with ThreadPoolExecutor(max_workers=len(section_assignments)) as ex:
    futures = {{
        ex.submit(run_subagent, client, COLLECTION_ID, LLM,
                  SUBAGENT_SYSTEM, f"Process {{r}} -- topic: {{t}}",
                  AGENT_TOOLS): r
        for r, t in section_assignments
    }}
    for fut in as_completed(futures):
        reply = fut.result()
```"""

_SDK_WORKSPACE_FILES = [
    ("h2ogpte-sdk/SKILL.md", "H2OGPTE Python SDK reference"),
    ("h2ogpte-sdk/references/patterns.md", "Sub-agent patterns (fan-out, fan-in, parallel)"),
    ("h2ogpte-sdk/references/client-api.md", "h2ogpte Python SDK -- full API reference"),
    ("h2ogpte-sdk/references/subagent.md", "Sub-agent spawning: `run_subagent`, fan-out pattern"),
    ("h2ogpte-sdk/scripts/fork_session.py", "Fork/clone a chat session (for parallel sub-agents)"),
    ("h2ogpte-sdk/scripts/subagent.py", "`run_subagent` -- spawn child agents in the same collection"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_system_prompt(
    *,
    role: str,
    skill_zip_name: str,
    skill_zip_contents: str,
    bootstrap_bash: str,
    workspace_files: list[tuple[str, str]],
    domain_guide: str,
    output_filename: str,
    output_instructions: str,
    key_rules: list[str],
    subagent_example: str = "",
    deps_path: str = "deps/",
    include_subagents: bool = True,
    include_sdk: bool = True,
    extra_sections: list[str] | None = None,
) -> str:
    """Build a complete agent system prompt from skill-specific parameters.

    Parameters
    ----------
    role:
        Opening sentence(s) describing the agent's expertise.
    skill_zip_name:
        Name of the primary skill zip uploaded to the collection.
    skill_zip_contents:
        Human-readable summary of what's inside the skill zip.
    bootstrap_bash:
        Complete bash block for bootstrapping the workspace *after* the
        two ``unzip`` commands.  Typically ``pip install`` lines and any
        extra setup.  The ``unzip`` commands are prepended automatically.
    workspace_files:
        List of ``(path, purpose)`` tuples describing files available
        after bootstrap.  The h2ogpte-sdk files are appended automatically.
    domain_guide:
        Arbitrary markdown with skill-specific workflows, best practices,
        critical warnings, etc.
    output_filename:
        Name of the final deliverable file.
    output_instructions:
        Markdown describing how to verify/validate the output.
    key_rules:
        List of one-line rules rendered as bold bullet points at the end.
    subagent_example:
        Optional custom sub-agent code example.  If empty, a generic
        example is used.
    deps_path:
        Path (relative to workspace) where Python wheels live.
        Default: ``"deps/"``
    include_subagents:
        Whether to include the sub-agents section.  Default: ``True``.
    include_sdk:
        Whether to include h2ogpte-sdk unzip command and workspace files.
        Set to ``False`` when h2ogpte-sdk.zip is not uploaded.  Default: ``True``.
    extra_sections:
        Optional list of additional markdown sections to append before
        the key rules.
    """
    # --- Workspace bootstrap ---
    sdk_unzip = (
        "unzip -o h2ogpte-sdk.zip -d h2ogpte-sdk/  # -> h2ogpte-sdk/SKILL.md, h2ogpte-sdk/references/, h2ogpte-sdk/scripts/\n"
        if include_sdk else ""
    )
    bootstrap = (
        "## Your Workspace\n"
        "\n"
        "When you start, your sandbox contains uploaded files. Bootstrap your workspace first:\n"
        "\n"
        "```bash\n"
        "# 1. Unzip skill resources (do this before anything else)\n"
        f"unzip -o {skill_zip_name}                      # -> {skill_zip_contents}\n"
        f"{sdk_unzip}"
        "\n"
        f"{bootstrap_bash}\n"
        "```"
    )

    # --- Workspace file table ---
    all_files = list(workspace_files) + (_SDK_WORKSPACE_FILES if include_sdk else [])
    table_rows = "\n".join(f"| `{path}` | {purpose} |" for path, purpose in all_files)
    workspace_table = (
        "After unzipping you have:\n"
        "| Path | Purpose |\n"
        "|------|---------|\n"
        f"{table_rows}"
    )

    # --- Key rules ---
    rules_md = "## Key Rules\n\n" + "\n".join(
        f"- **{rule}**" for rule in key_rules
    )

    # --- Output section ---
    output_section = (
        "## Output Requirements\n"
        "\n"
        f"1. Save your final output as `{output_filename}` in the current working directory\n"
        f"2. {output_instructions}\n"
        f"3. When complete, write `DONE: {output_filename}`"
    )

    # --- Assemble ---
    sections = [
        role,
        "",
        _CODE_EXECUTION,
        "",
        bootstrap,
        "",
        workspace_table,
        "",
        domain_guide,
        "",
    ]

    if include_subagents:
        subagent_text = subagent_example if subagent_example else _SUBAGENT_SECTION
        sections.extend([subagent_text, ""])

    if extra_sections:
        for s in extra_sections:
            sections.extend([s, ""])

    sections.extend([
        output_section,
        "",
        rules_md,
        "",
        _NO_INTERNET.format(
            skill_zip_name=skill_zip_name,
            deps_path=deps_path,
            sdk_unzip_line="unzip -o h2ogpte-sdk.zip -d h2ogpte-sdk/\n" if include_sdk else "",
        ),
    ])

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(
    *,
    request: str,
    file_names: list[str],
    collection_id: str,
    h2ogpte_url: str,
    api_key: str,
    llm: str,
    agent_total_timeout: int,
    project_name: str,
    output_filename: str,
    output_verification_bash: str = "",
    output_verification_note: str = "",
    agent_tools: list[str] | None = None,
    openai_agent_base_url: str = "",
    extra_user_sections: str = "",
) -> str:
    """Build the user-facing prompt injected as the first message.

    Parameters
    ----------
    request:
        The user's task description.
    file_names:
        List of filenames already uploaded to the workspace.
    collection_id:
        h2ogpte collection ID (for sub-agent connection values).
    h2ogpte_url:
        h2ogpte server address.
    api_key:
        h2ogpte API key.
    llm:
        LLM model ID.
    agent_total_timeout:
        Server-side agent total timeout in seconds.
    project_name:
        Short slug for the project.
    output_filename:
        Expected output filename.
    output_verification_bash:
        Optional bash command to run after building.
    output_verification_note:
        Optional explanation of verification exit codes / results.
    agent_tools:
        List of agent tool names.  Default includes python, shell,
        rag_text, rag_vision, and claude_tool_runner (for MCP access).
    openai_agent_base_url:
        Optional OpenAI-compatible base URL for sub-agent spawning.
    extra_user_sections:
        Optional extra markdown to append.
    """
    if agent_tools is None:
        agent_tools = [
            "python", "shell", "rag_text", "rag_vision",
            "convert_document_to_text.py", "ask_question_about_documents.py",
            "mermaid_renderer.py", "claude_tool_runner.py",
        ]

    files_section = ""
    if file_names:
        files_section = (
            "\n## Input Files Available\n\n"
            + "\n".join(f"- `{f}` (uploaded to your workspace)" for f in file_names)
            + "\n\nThese files are already in your agent sandbox -- no download needed.\n"
        )

    verification = ""
    if output_verification_bash:
        verification = f"\n```bash\n{output_verification_bash}\n```\n"
        if output_verification_note:
            verification += f"\n{output_verification_note}\n"

    qa_config = ""
    if openai_agent_base_url:
        qa_config = (
            "\n**Run this immediately after bootstrap to save credentials for tools that need them:**\n"
            "```python\n"
            "import json\n"
            f'json.dump({{"h2ogpte_address": "{h2ogpte_url}", "h2ogpte_api_key": "{api_key}",\n'
            f'           "llm": "{llm}", "openai_base_url": "{openai_agent_base_url}",\n'
            '           "h2ogpte_verify": False},\n'
            '          open(".qa_config.json", "w"))\n'
            "```"
        )

    return f"""\
## Your Task

{request}
{files_section}{extra_user_sections}
## Connection Values (for sub-agents)

```
H2OGPTE_ADDRESS      = "{h2ogpte_url}"
H2OGPTE_API_KEY      = "{api_key}"
COLLECTION_ID        = "{collection_id}"
LLM                  = "{llm}"
PROJECT_NAME         = "{project_name}"
AGENT_TOOLS          = {agent_tools!r}
MAX_SUBAGENT_TIMEOUT = {agent_total_timeout}
```

Use these exact values when spawning sub-agents -- do not read from environment variables (sandbox has none).
{qa_config}
## Output

Save your final output as `{output_filename}` in the current working directory.
{verification}
**IMPORTANT:** When fixing issues, always edit your source code and rebuild from scratch. Never manually patch the output file -- always regenerate from source.

When the output is complete and verified, write `DONE: {output_filename}` and stop. The file is the only deliverable.
"""


# ---------------------------------------------------------------------------
# Default prompt — dependency bootstrap only (run this file to print it)
# ---------------------------------------------------------------------------

def build_sumbench_system_prompt() -> str:
    """Build the default SumBench MCP system prompt.

    Two-part deployment: the MCP server code is uploaded as a Local MCP Tool
    (sumbench_mcp.zip), while deps.zip with Python wheels is uploaded to the
    collection.  This prompt tells the agent to install deps before calling
    any MCP tools.
    """
    return build_system_prompt(
        role=(
            "You are an expert summary evaluation agent. You evaluate "
            "LLM-generated summaries using the H2O SumBench MCP tools. "
            "You run comprehensive multi-metric evaluation and present "
            "clear, actionable results."
        ),
        skill_zip_name="deps.zip",
        skill_zip_contents="deps/ (Python wheel files), requirements.txt, nltk_data/",
        bootstrap_bash=(
            "# 2. Install MCP server dependencies from local wheels\n"
            "pip install --quiet --no-deps deps/*.whl"
        ),
        workspace_files=[
            ("deps/", "Python wheel files for offline installation"),
            ("requirements.txt", "Python dependencies for the MCP server"),
            ("nltk_data/", "Bundled NLTK data (punkt_tab, wordnet) for METEOR"),
        ],
        domain_guide=(
            "## CRITICAL: Set Environment Variables (Python)\n\n"
            "The sandbox **blocks shell `export`**. Set all environment variables "
            "via Python **before** calling any MCP tools:\n\n"
            "```python\n"
            "import os\n"
            "os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), 'nltk_data')\n"
            "os.environ['SUMBENCH_AIRGAPPED'] = '1'\n"
            "os.environ['HF_HUB_OFFLINE'] = '1'\n"
            "os.environ['TRANSFORMERS_OFFLINE'] = '1'\n"
            "print('Environment configured')\n"
            "```\n\n"
            "**Do NOT set H2OGPTE_API_KEY or H2OGPTE_ADDRESS via os.environ.** "
            "These credentials are injected automatically by the platform as encrypted "
            "secrets. Setting them manually will cause a decryption error.\n\n"
            "## How to Call MCP Tools\n\n"
            "Import `claude_tool_runner` and call it with a natural-language query. "
            "This is the **ONLY** way to invoke the evaluation.\n\n"
            "**DO NOT** import from `mcp_tools_runner`, `evaluators`, `tool_logic`, "
            "or `server.py` directly. DO NOT inspect module contents or function "
            "signatures. The MCP server handles everything internally.\n\n"
            "### Step 1: Verify Environment\n\n"
            "After bootstrap, call `check_environment` to verify everything is ready:\n\n"
            "```python\n"
            "from api_server.agent_tools.claude_tool_runner import claude_tool_runner\n"
            "result = claude_tool_runner(query=\"Call the check_environment tool to verify "
            "the MCP server environment is ready. Report the full status.\")\n"
            "print(result)\n"
            "```\n\n"
            "If any component shows FAILED or MISSING, fix it before proceeding.\n\n"
            "### Step 2: Run Evaluation\n\n"
            "```python\n"
            "from api_server.agent_tools.claude_tool_runner import claude_tool_runner\n"
            "\n"
            "query = \"\"\"\n"
            "Evaluate the quality of this generated summary using the Sumbench MCP tool.\n"
            "\n"
            "**Generated Summary:**\n"
            "<paste full summary text here>\n"
            "\n"
            "**Source Text:**\n"
            "<paste full source text here>\n"
            "\n"
            "**Reference Summary:**\n"
            "<paste full reference text here>\n"
            "\n"
            "This is a <scenario name> scenario. "
            "Use the run_multiple tool with these metrics:\n"
            "- Metrics: <metric list for this scenario>\n"
            "- Parameters: summary, source, reference (as applicable)\n"
            "\n"
            "Provide the complete metric results.\n"
            "\"\"\"\n"
            "\n"
            "result = claude_tool_runner(query=query)\n"
            "```\n\n"
            "### Scenario Detection & Metric Selection\n\n"
            "First, identify which inputs are available, then call `run_multiple` "
            "with the correct metrics for that scenario.\n\n"
            "This is an **airgapped** environment — metrics requiring model downloads "
            "(perplexity, bertscore, entity_coverage, semantic_coverage, bertscore_recall) "
            "are NOT available. Use only the metrics listed below.\n\n"
            "**Scenario A — Source + Reference (Full Diagnostic):**\n"
            "- Metrics: `[\"rouge\", \"bleu\", \"meteor\", \"levenshtein\", \"chrf\", "
            "\"factchecker_api\", \"llm_faithfulness\", \"llm_coherence\", "
            "\"llm_relevance\", \"llm_fluency\", \"llm_dag\", \"llm_prometheus\"]`\n"
            "- Parameters: summary, source, reference\n\n"
            "**Scenario B — Source Only (Truth-First):**\n"
            "- Metrics: `[\"factchecker_api\", \"llm_faithfulness\", \"llm_relevance\"]`\n"
            "- Parameters: summary, source\n\n"
            "**Scenario C — Reference Only (Stylistic-Match):**\n"
            "- Metrics: `[\"rouge\", \"bleu\", \"meteor\", \"levenshtein\", \"chrf\", "
            "\"llm_coherence\"]`\n"
            "- Parameters: summary, reference\n\n"
            "**Scenario D — Neither (Linguistic-Sanity):**\n"
            "- Metrics: `[\"llm_fluency\"]`\n"
            "- Parameters: summary\n\n"
            "**IMPORTANT:**\n"
            "- Always install dependencies and set env vars BEFORE calling claude_tool_runner\n"
            "- Call run_multiple ONCE with all applicable metrics — do not call metrics individually\n"
            "- Do NOT try to call evaluation functions directly — use claude_tool_runner ONLY"
        ),
        output_filename="evaluation_report.md",
        output_instructions="Verify the report contains a score table and overall assessment",
        key_rules=[
            "Install dependencies from deps.zip BEFORE calling any MCP tools",
            "Call check_environment via claude_tool_runner AFTER bootstrap to verify NLTK data and H2OGPTE credentials before evaluating",
            "Do NOT use shell `export` — it is blocked. Use Python `os.environ` for NLTK_DATA and airgapped flags only",
            "Do NOT set H2OGPTE_API_KEY or H2OGPTE_ADDRESS via os.environ — credentials are injected automatically by the platform as encrypted secrets",
            "NEVER import from mcp_tools_runner, evaluators, tool_logic, or server.py — use claude_tool_runner ONLY",
            "Detect the scenario (source+reference, source only, reference only, neither) and select metrics accordingly",
            "Call run_multiple ONCE via claude_tool_runner with the correct airgapped metric list for the scenario",
            "Always pass FULL text in the query, never filenames or URLs",
            "Use claude_tool_runner(query=...) — NOT litellm_tool_runner",
            "Present results as a Markdown table: Category | Metric | Score | Interpretation",
            "Include 3–4 bullet-point insights and an overall quality assessment",
        ],
        include_subagents=False,
        include_sdk=False,
    )


def build_sumbench_user_prompt(
    *,
    request: str,
    file_names: list[str] | None = None,
    collection_id: str = "<COLLECTION_ID>",
    h2ogpte_url: str = "<H2OGPTE_ADDRESS>",
    api_key: str = "<H2OGPTE_API_KEY>",
    llm: str = "<LLM>",
    agent_total_timeout: int = 600,
) -> str:
    """Build the user prompt for a SumBench evaluation task.

    All connection values default to placeholders — replace them with
    real values before pasting into the H2OGPTe UI.
    """
    return build_user_prompt(
        request=request,
        file_names=file_names or [],
        collection_id=collection_id,
        h2ogpte_url=h2ogpte_url,
        api_key=api_key,
        llm=llm,
        agent_total_timeout=agent_total_timeout,
        project_name="sumbench-eval",
        output_filename="evaluation_report.md",
        agent_tools=[
            "python", "shell", "rag_text",
            "claude_tool_runner.py",
        ],
    )


if __name__ == "__main__":
    print("=" * 72)
    print("SYSTEM PROMPT")
    print("=" * 72)
    print(build_sumbench_system_prompt())
    print()
    print("=" * 72)
    print()
    print()
    print("=" * 72)
    print("USER PROMPT (replace <PLACEHOLDERS> with real values)")
    print("=" * 72)
    print(build_sumbench_user_prompt(
        request=(
            "Bootstrap your workspace first (unzip deps, install packages), "
            "then evaluate the following summary using the evaluate_summary "
            "MCP tool.\n\n"
            "**Summary:** <paste summary here>\n\n"
            "**Source:** <paste source document here>"
        ),
    ))
    print("=" * 72)
