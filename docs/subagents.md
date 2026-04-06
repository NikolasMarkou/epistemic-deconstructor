# Claude Code Subagents - Complete Reference

> Sources: [Create custom subagents](https://code.claude.com/docs/en/sub-agents) | [Agent Teams](https://code.claude.com/docs/en/agent-teams) | [Subagents in the SDK](https://docs.anthropic.com/en/docs/claude-code/sdk/subagents)

## Table of Contents

- [Overview](#overview)
- [Built-in Subagents](#built-in-subagents)
- [Quickstart: Create Your First Subagent](#quickstart-create-your-first-subagent)
- [Configure Subagents](#configure-subagents)
  - [/agents Command](#use-the-agents-command)
  - [Subagent Scope & Priority](#choose-the-subagent-scope)
  - [Write Subagent Files](#write-subagent-files)
  - [Supported Frontmatter Fields](#supported-frontmatter-fields)
  - [Choose a Model](#choose-a-model)
  - [Control Capabilities](#control-subagent-capabilities)
  - [Permission Modes](#permission-modes)
  - [Preload Skills](#preload-skills-into-subagents)
  - [Persistent Memory](#enable-persistent-memory)
  - [Hooks](#define-hooks-for-subagents)
  - [Disable Specific Subagents](#disable-specific-subagents)
- [Work with Subagents](#work-with-subagents)
  - [Automatic Delegation](#understand-automatic-delegation)
  - [Explicit Invocation](#invoke-subagents-explicitly)
  - [Foreground vs Background](#run-subagents-in-foreground-or-background)
  - [Common Patterns](#common-patterns)
  - [Main Conversation vs Subagent](#choose-between-subagents-and-main-conversation)
  - [Context Management](#manage-subagent-context)
- [Example Subagents](#example-subagents)
- [Subagents in the Agent SDK (Programmatic)](#subagents-in-the-agent-sdk)
  - [Programmatic Definition](#programmatic-definition)
  - [AgentDefinition Configuration](#agentdefinition-configuration)
  - [What Subagents Inherit](#what-subagents-inherit)
  - [Invoking Subagents](#invoking-subagents-in-the-sdk)
  - [Detecting Subagent Invocation](#detecting-subagent-invocation)
  - [Resuming Subagents (SDK)](#resuming-subagents-sdk)
  - [Tool Restrictions (SDK)](#tool-restrictions-sdk)
  - [Common Tool Combinations](#common-tool-combinations)
  - [Troubleshooting (SDK)](#troubleshooting-sdk)
- [Agent Teams](#agent-teams)
  - [When to Use Agent Teams](#when-to-use-agent-teams)
  - [Subagents vs Agent Teams](#compare-with-subagents)
  - [Enable Agent Teams](#enable-agent-teams)
  - [Start a Team](#start-your-first-agent-team)
  - [Control Your Team](#control-your-agent-team)
  - [How Agent Teams Work](#how-agent-teams-work)
  - [Use Case Examples](#use-case-examples)
  - [Best Practices (Teams)](#best-practices-teams)
  - [Troubleshooting (Teams)](#troubleshooting-teams)
  - [Limitations](#limitations)

---

## Overview

Subagents are specialized AI assistants that handle specific types of tasks. Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions. When Claude encounters a task that matches a subagent's description, it delegates to that subagent, which works independently and returns results.

> **Note:** If you need multiple agents working in parallel and communicating with each other, see [Agent Teams](#agent-teams) instead. Subagents work within a single session; agent teams coordinate across separate sessions.

Subagents help you:

- **Preserve context** by keeping exploration and implementation out of your main conversation
- **Enforce constraints** by limiting which tools a subagent can use
- **Reuse configurations** across projects with user-level subagents
- **Specialize behavior** with focused system prompts for specific domains
- **Control costs** by routing tasks to faster, cheaper models like Haiku

Claude uses each subagent's description to decide when to delegate tasks. When you create a subagent, write a clear description so Claude knows when to use it.

---

## Built-in Subagents

Claude Code includes built-in subagents that Claude automatically uses when appropriate. Each inherits the parent conversation's permissions with additional tool restrictions.

### Explore

A fast, read-only agent optimized for searching and analyzing codebases.

- **Model**: Haiku (fast, low-latency)
- **Tools**: Read-only tools (denied access to Write and Edit tools)
- **Purpose**: File discovery, code search, codebase exploration

Claude delegates to Explore when it needs to search or understand a codebase without making changes. When invoking Explore, Claude specifies a thoroughness level: **quick** for targeted lookups, **medium** for balanced exploration, or **very thorough** for comprehensive analysis.

### Plan

A research agent used during plan mode to gather context before presenting a plan.

- **Model**: Inherits from main conversation
- **Tools**: Read-only tools (denied access to Write and Edit tools)
- **Purpose**: Codebase research for planning

When you're in plan mode and Claude needs to understand your codebase, it delegates research to the Plan subagent. This prevents infinite nesting (subagents cannot spawn other subagents) while still gathering necessary context.

### General-purpose

A capable agent for complex, multi-step tasks that require both exploration and action.

- **Model**: Inherits from main conversation
- **Tools**: All tools
- **Purpose**: Complex research, multi-step operations, code modifications

Claude delegates to general-purpose when the task requires both exploration and modification, complex reasoning to interpret results, or multiple dependent steps.

### Other Helper Agents

| Agent             | Model  | When Claude uses it                                      |
| :---------------- | :----- | :------------------------------------------------------- |
| statusline-setup  | Sonnet | When you run `/statusline` to configure your status line |
| Claude Code Guide | Haiku  | When you ask questions about Claude Code features        |

---

## Quickstart: Create Your First Subagent

Subagents are defined in Markdown files with YAML frontmatter. You can create them manually or use the `/agents` command.

### Walkthrough with `/agents`

1. **Open the subagents interface**: Run `/agents` in Claude Code
2. **Choose a location**: Select **Create new agent**, then choose **Personal** (saves to `~/.claude/agents/`)
3. **Generate with Claude**: Select **Generate with Claude** and describe the subagent:
   ```
   A code improvement agent that scans files and suggests improvements
   for readability, performance, and best practices. It should explain
   each issue, show the current code, and provide an improved version.
   ```
4. **Select tools**: For a read-only reviewer, deselect everything except **Read-only tools**
5. **Select model**: Choose which model the subagent uses (e.g. **Sonnet**)
6. **Choose a color**: Pick a background color for the subagent in the UI
7. **Configure memory**: Select **User scope** for persistent memory at `~/.claude/agent-memory/`, or **None**
8. **Save and try it out**: Press `s` or `Enter` to save. Try it:
   ```
   Use the code-improver agent to suggest improvements in this project
   ```

---

## Configure Subagents

### Use the /agents Command

The `/agents` command provides an interactive interface for managing subagents:

- View all available subagents (built-in, user, project, and plugin)
- Create new subagents with guided setup or Claude generation
- Edit existing subagent configuration and tool access
- Delete custom subagents
- See which subagents are active when duplicates exist

To list all configured subagents from the command line without starting an interactive session, run `claude agents`.

### Choose the Subagent Scope

Subagents are Markdown files with YAML frontmatter. Store them in different locations depending on scope. When multiple subagents share the same name, the higher-priority location wins.

| Location                     | Scope                   | Priority    | How to create                         |
| :--------------------------- | :---------------------- | :---------- | :------------------------------------ |
| Managed settings             | Organization-wide       | 1 (highest) | Deployed via managed settings         |
| `--agents` CLI flag          | Current session         | 2           | Pass JSON when launching Claude Code  |
| `.claude/agents/`            | Current project         | 3           | Interactive or manual                 |
| `~/.claude/agents/`          | All your projects       | 4           | Interactive or manual                 |
| Plugin's `agents/` directory | Where plugin is enabled | 5 (lowest)  | Installed with plugins                |

**Project subagents** (`.claude/agents/`) are ideal for subagents specific to a codebase. Check them into version control so your team can use and improve them collaboratively.

**User subagents** (`~/.claude/agents/`) are personal subagents available in all your projects.

**CLI-defined subagents** are passed as JSON when launching Claude Code. They exist only for that session:

```bash
claude --agents '{
  "code-reviewer": {
    "description": "Expert code reviewer. Use proactively after code changes.",
    "prompt": "You are a senior code reviewer. Focus on code quality, security, and best practices.",
    "tools": ["Read", "Grep", "Glob", "Bash"],
    "model": "sonnet"
  },
  "debugger": {
    "description": "Debugging specialist for errors and test failures.",
    "prompt": "You are an expert debugger. Analyze errors, identify root causes, and provide fixes."
  }
}'
```

The `--agents` flag accepts JSON with the same frontmatter fields as file-based subagents: `description`, `prompt`, `tools`, `disallowedTools`, `model`, `permissionMode`, `mcpServers`, `hooks`, `maxTurns`, `skills`, `initialPrompt`, `memory`, `effort`, `background`, `isolation`, and `color`.

**Plugin subagents** come from plugins you've installed. For security reasons, plugin subagents do not support the `hooks`, `mcpServers`, or `permissionMode` frontmatter fields.

### Write Subagent Files

Subagent files use YAML frontmatter for configuration, followed by the system prompt in Markdown:

> **Note**: Subagents are loaded at session start. If you create a subagent by manually adding a file, restart your session or use `/agents` to load it immediately.

```markdown
---
name: code-reviewer
description: Reviews code for quality and best practices
tools: Read, Glob, Grep
model: sonnet
---

You are a code reviewer. When invoked, analyze the code and provide
specific, actionable feedback on quality, security, and best practices.
```

The frontmatter defines the subagent's metadata and configuration. The body becomes the system prompt. Subagents receive only this system prompt (plus basic environment details), not the full Claude Code system prompt.

### Supported Frontmatter Fields

Only `name` and `description` are required.

| Field             | Required | Description                                                                                                            |
| :---------------- | :------- | :--------------------------------------------------------------------------------------------------------------------- |
| `name`            | Yes      | Unique identifier using lowercase letters and hyphens                                                                  |
| `description`     | Yes      | When Claude should delegate to this subagent                                                                           |
| `tools`           | No       | Tools the subagent can use. Inherits all tools if omitted                                                              |
| `disallowedTools` | No       | Tools to deny, removed from inherited or specified list                                                                |
| `model`           | No       | Model to use: `sonnet`, `opus`, `haiku`, a full model ID (e.g. `claude-opus-4-6`), or `inherit`. Defaults to `inherit` |
| `permissionMode`  | No       | Permission mode: `default`, `acceptEdits`, `auto`, `dontAsk`, `bypassPermissions`, or `plan`                          |
| `maxTurns`        | No       | Maximum number of agentic turns before the subagent stops                                                              |
| `skills`          | No       | Skills to load into the subagent's context at startup (full content injected, not just available for invocation)        |
| `mcpServers`      | No       | MCP servers available to this subagent (string references or inline definitions)                                       |
| `hooks`           | No       | Lifecycle hooks scoped to this subagent                                                                                |
| `memory`          | No       | Persistent memory scope: `user`, `project`, or `local`                                                                 |
| `background`      | No       | Set to `true` to always run as a background task. Default: `false`                                                     |
| `effort`          | No       | Effort level: `low`, `medium`, `high`, `max` (Opus 4.6 only). Default: inherits from session                          |
| `isolation`       | No       | Set to `worktree` to run in a temporary git worktree (isolated repo copy, auto-cleaned if no changes)                  |
| `color`           | No       | Display color: `red`, `blue`, `green`, `yellow`, `purple`, `orange`, `pink`, or `cyan`                                 |
| `initialPrompt`   | No       | Auto-submitted as first user turn when running as main session agent (via `--agent`). Commands and skills are processed |

### Choose a Model

The `model` field controls which AI model the subagent uses:

- **Model alias**: `sonnet`, `opus`, or `haiku`
- **Full model ID**: e.g. `claude-opus-4-6` or `claude-sonnet-4-6`
- **inherit**: Use the same model as the main conversation
- **Omitted**: Defaults to `inherit`

Model resolution order:

1. The `CLAUDE_CODE_SUBAGENT_MODEL` environment variable, if set
2. The per-invocation `model` parameter
3. The subagent definition's `model` frontmatter
4. The main conversation's model

### Control Subagent Capabilities

#### Available Tools

By default, subagents inherit all tools from the main conversation. To restrict tools, use `tools` (allowlist) or `disallowedTools` (denylist):

```yaml
---
name: safe-researcher
description: Research agent with restricted capabilities
tools: Read, Grep, Glob, Bash
---
```

```yaml
---
name: no-writes
description: Inherits every tool except file writes
disallowedTools: Write, Edit
---
```

If both are set, `disallowedTools` is applied first, then `tools` is resolved against the remaining pool.

#### Restrict Which Subagents Can Be Spawned

When an agent runs as the main thread with `claude --agent`, use `Agent(agent_type)` syntax to restrict which subagent types it can spawn:

> **Note**: In version 2.1.63, the Task tool was renamed to Agent. Existing `Task(...)` references still work as aliases.

```yaml
---
name: coordinator
description: Coordinates work across specialized agents
tools: Agent(worker, researcher), Read, Bash
---
```

This is an allowlist: only the `worker` and `researcher` subagents can be spawned. To allow spawning any subagent, use `Agent` without parentheses. If `Agent` is omitted from `tools` entirely, the agent cannot spawn any subagents.

This restriction only applies to agents running as the main thread with `claude --agent`. Subagents cannot spawn other subagents.

#### Scope MCP Servers to a Subagent

Use `mcpServers` to give a subagent access to MCP servers:

```yaml
---
name: browser-tester
description: Tests features in a real browser using Playwright
mcpServers:
  # Inline definition: scoped to this subagent only
  - playwright:
      type: stdio
      command: npx
      args: ["-y", "@playwright/mcp@latest"]
  # Reference by name: reuses an already-configured server
  - github
---

Use the Playwright tools to navigate, screenshot, and interact with pages.
```

Inline definitions use the same schema as `.mcp.json` server entries (`stdio`, `http`, `sse`, `ws`), keyed by the server name. To keep an MCP server out of the main conversation entirely, define it inline here.

### Permission Modes

| Mode                | Behavior                                                                    |
| :------------------ | :-------------------------------------------------------------------------- |
| `default`           | Standard permission checking with prompts                                   |
| `acceptEdits`       | Auto-accept file edits except in protected directories                      |
| `auto`              | Background classifier reviews commands and protected-directory writes       |
| `dontAsk`           | Auto-deny permission prompts (explicitly allowed tools still work)          |
| `bypassPermissions` | Skip permission prompts (use with caution)                                  |
| `plan`              | Plan mode (read-only exploration)                                           |

> **Warning**: `bypassPermissions` skips permission prompts. Writes to `.git`, `.claude`, `.vscode`, `.idea`, and `.husky` directories still prompt.

If the parent uses `bypassPermissions`, this takes precedence and cannot be overridden. If the parent uses auto mode, the subagent inherits it and any `permissionMode` in frontmatter is ignored.

### Preload Skills into Subagents

```yaml
---
name: api-developer
description: Implement API endpoints following team conventions
skills:
  - api-conventions
  - error-handling-patterns
---

Implement API endpoints. Follow the conventions and patterns from the preloaded skills.
```

The full content of each skill is injected into the subagent's context, not just made available for invocation. Subagents don't inherit skills from the parent conversation; you must list them explicitly.

### Enable Persistent Memory

The `memory` field gives the subagent a persistent directory that survives across conversations:

```yaml
---
name: code-reviewer
description: Reviews code for quality and best practices
memory: user
---

You are a code reviewer. As you review code, update your agent memory with
patterns, conventions, and recurring issues you discover.
```

| Scope     | Location                                      | Use when                                                       |
| :-------- | :-------------------------------------------- | :------------------------------------------------------------- |
| `user`    | `~/.claude/agent-memory/<name-of-agent>/`     | Learnings should persist across all projects                   |
| `project` | `.claude/agent-memory/<name-of-agent>/`       | Knowledge is project-specific and shareable via version control |
| `local`   | `.claude/agent-memory-local/<name-of-agent>/` | Knowledge is project-specific but should NOT be committed      |

When memory is enabled:
- The subagent's system prompt includes instructions for reading/writing to the memory directory
- First 200 lines or 25KB of `MEMORY.md` is included in the system prompt
- Read, Write, and Edit tools are automatically enabled

**Tips:**
- `project` is the recommended default scope
- Ask the subagent to consult its memory before starting work
- Ask the subagent to update its memory after completing a task
- Include memory instructions directly in the subagent's markdown file

### Define Hooks for Subagents

#### Hooks in Subagent Frontmatter

Define hooks directly in the subagent's markdown file. These only run while that specific subagent is active:

| Event         | Matcher input | When it fires                                                       |
| :------------ | :------------ | :------------------------------------------------------------------ |
| `PreToolUse`  | Tool name     | Before the subagent uses a tool                                     |
| `PostToolUse` | Tool name     | After the subagent uses a tool                                      |
| `Stop`        | (none)        | When the subagent finishes (converted to `SubagentStop` at runtime) |

```yaml
---
name: code-reviewer
description: Review code changes with automatic linting
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-command.sh $TOOL_INPUT"
  PostToolUse:
    - matcher: "Edit|Write"
      hooks:
        - type: command
          command: "./scripts/run-linter.sh"
---
```

#### Project-level Hooks for Subagent Events

Configure hooks in `settings.json` that respond to subagent lifecycle events:

| Event           | Matcher input   | When it fires                    |
| :-------------- | :-------------- | :------------------------------- |
| `SubagentStart` | Agent type name | When a subagent begins execution |
| `SubagentStop`  | Agent type name | When a subagent completes        |

```json
{
  "hooks": {
    "SubagentStart": [
      {
        "matcher": "db-agent",
        "hooks": [
          { "type": "command", "command": "./scripts/setup-db-connection.sh" }
        ]
      }
    ],
    "SubagentStop": [
      {
        "hooks": [
          { "type": "command", "command": "./scripts/cleanup-db-connection.sh" }
        ]
      }
    ]
  }
}
```

### Disable Specific Subagents

Add to the `deny` array in your settings using `Agent(subagent-name)`:

```json
{
  "permissions": {
    "deny": ["Agent(Explore)", "Agent(my-custom-agent)"]
  }
}
```

Or via CLI flag:

```bash
claude --disallowedTools "Agent(Explore)"
```

---

## Work with Subagents

### Understand Automatic Delegation

Claude automatically delegates based on the task description, `description` field in subagent configs, and current context. To encourage proactive delegation, include phrases like "use proactively" in your subagent's description.

### Invoke Subagents Explicitly

Three patterns, escalating from one-off to session-wide:

**Natural language** - name the subagent in your prompt:
```
Use the test-runner subagent to fix failing tests
Have the code-reviewer subagent look at my recent changes
```

**@-mention** - guarantees the subagent runs for one task:
```
@"code-reviewer (agent)" look at the auth changes
```

Your full message still goes to Claude, which writes the subagent's task prompt. The @-mention controls *which* subagent runs.

**Session-wide** - run the whole session as a subagent:

```bash
claude --agent code-reviewer
```

The subagent's system prompt replaces the default Claude Code system prompt entirely. `CLAUDE.md` files and project memory still load. The agent name appears as `@<name>` in the startup header.

To make it the default for every session in a project:

```json
{
  "agent": "code-reviewer"
}
```

### Run Subagents in Foreground or Background

- **Foreground subagents** block the main conversation until complete. Permission prompts and clarifying questions are passed through.
- **Background subagents** run concurrently. Claude Code prompts for tool permissions upfront. Once running, the subagent auto-denies anything not pre-approved. Clarifying questions fail but the subagent continues.

You can:
- Ask Claude to "run this in the background"
- Press **Ctrl+B** to background a running task
- Set `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` to disable all background tasks

### Common Patterns

#### Isolate High-Volume Operations

```
Use a subagent to run the test suite and report only the failing tests with their error messages
```

#### Run Parallel Research

```
Research the authentication, database, and API modules in parallel using separate subagents
```

> **Warning**: Running many subagents that each return detailed results can consume significant context.

#### Chain Subagents

```
Use the code-reviewer subagent to find performance issues, then use the optimizer subagent to fix them
```

### Choose Between Subagents and Main Conversation

Use the **main conversation** when:
- The task needs frequent back-and-forth or iterative refinement
- Multiple phases share significant context
- You're making a quick, targeted change
- Latency matters (subagents start fresh)

Use **subagents** when:
- The task produces verbose output you don't need in your main context
- You want to enforce specific tool restrictions or permissions
- The work is self-contained and can return a summary

> **Note**: Subagents cannot spawn other subagents. If your workflow requires nested delegation, use Skills or chain subagents from the main conversation.

For a quick question about something already in your conversation, use `/btw` instead of a subagent. It sees your full context but has no tool access, and the answer is discarded.

### Manage Subagent Context

#### Resume Subagents

Each subagent invocation creates a new instance with fresh context. To continue an existing subagent's work:

```
Use the code-reviewer subagent to review the authentication module
[Agent completes]

Continue that code review and now analyze the authorization logic
[Claude resumes the subagent with full context from previous conversation]
```

If a stopped subagent receives a `SendMessage`, it auto-resumes in the background without requiring a new `Agent` invocation.

Subagent transcripts persist at `~/.claude/projects/{project}/{sessionId}/subagents/` as `agent-{agentId}.jsonl`.

Subagent transcripts persist independently:
- **Main conversation compaction**: subagent transcripts are unaffected (stored separately)
- **Session persistence**: subagent transcripts persist within their session
- **Automatic cleanup**: based on `cleanupPeriodDays` setting (default: 30 days)

#### Auto-compaction

Subagents support automatic compaction (triggers at ~95% capacity). Set `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` to trigger earlier (e.g. `50`).

---

## Example Subagents

### Code Reviewer

```markdown
---
name: code-reviewer
description: Expert code review specialist. Proactively reviews code for quality, security, and maintainability. Use immediately after writing or modifying code.
tools: Read, Grep, Glob, Bash
model: inherit
---

You are a senior code reviewer ensuring high standards of code quality and security.

When invoked:
1. Run git diff to see recent changes
2. Focus on modified files
3. Begin review immediately

Review checklist:
- Code is clear and readable
- Functions and variables are well-named
- No duplicated code
- Proper error handling
- No exposed secrets or API keys
- Input validation implemented
- Good test coverage
- Performance considerations addressed

Provide feedback organized by priority:
- Critical issues (must fix)
- Warnings (should fix)
- Suggestions (consider improving)

Include specific examples of how to fix issues.
```

### Debugger

```markdown
---
name: debugger
description: Debugging specialist for errors, test failures, and unexpected behavior. Use proactively when encountering any issues.
tools: Read, Edit, Bash, Grep, Glob
---

You are an expert debugger specializing in root cause analysis.

When invoked:
1. Capture error message and stack trace
2. Identify reproduction steps
3. Isolate the failure location
4. Implement minimal fix
5. Verify solution works

Debugging process:
- Analyze error messages and logs
- Check recent code changes
- Form and test hypotheses
- Add strategic debug logging
- Inspect variable states

For each issue, provide:
- Root cause explanation
- Evidence supporting the diagnosis
- Specific code fix
- Testing approach
- Prevention recommendations

Focus on fixing the underlying issue, not the symptoms.
```

### Data Scientist

```markdown
---
name: data-scientist
description: Data analysis expert for SQL queries, BigQuery operations, and data insights. Use proactively for data analysis tasks and queries.
tools: Bash, Read, Write
model: sonnet
---

You are a data scientist specializing in SQL and BigQuery analysis.

When invoked:
1. Understand the data analysis requirement
2. Write efficient SQL queries
3. Use BigQuery command line tools (bq) when appropriate
4. Analyze and summarize results
5. Present findings clearly

Key practices:
- Write optimized SQL queries with proper filters
- Use appropriate aggregations and joins
- Include comments explaining complex logic
- Format results for readability
- Provide data-driven recommendations
```

### Database Query Validator (with Hook)

```markdown
---
name: db-reader
description: Execute read-only database queries. Use when analyzing data or generating reports.
tools: Bash
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-readonly-query.sh"
---

You are a database analyst with read-only access. Execute SELECT queries to answer questions about the data.
```

Validation script (`./scripts/validate-readonly-query.sh`):

```bash
#!/bin/bash
# Blocks SQL write operations, allows SELECT queries

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [ -z "$COMMAND" ]; then
  exit 0
fi

# Block write operations (case-insensitive)
if echo "$COMMAND" | grep -iE '\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE)\b' > /dev/null; then
  echo "Blocked: Write operations not allowed. Use SELECT queries only." >&2
  exit 2
fi

exit 0
```

---

## Subagents in the Agent SDK

### Programmatic Definition

Define subagents directly in code using the `agents` parameter. The `Agent` tool must be included in `allowedTools`.

**Python:**

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition


async def main():
    async for message in query(
        prompt="Review the authentication module for security issues",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Grep", "Glob", "Agent"],
            agents={
                "code-reviewer": AgentDefinition(
                    description="Expert code review specialist. Use for quality, security, and maintainability reviews.",
                    prompt="""You are a code review specialist with expertise in security, performance, and best practices.

When reviewing code:
- Identify security vulnerabilities
- Check for performance issues
- Verify adherence to coding standards
- Suggest specific improvements

Be thorough but concise in your feedback.""",
                    tools=["Read", "Grep", "Glob"],
                    model="sonnet",
                ),
                "test-runner": AgentDefinition(
                    description="Runs and analyzes test suites. Use for test execution and coverage analysis.",
                    prompt="""You are a test execution specialist. Run tests and provide clear analysis of results.

Focus on:
- Running test commands
- Analyzing test output
- Identifying failing tests
- Suggesting fixes for failures""",
                    tools=["Bash", "Read", "Grep"],
                ),
            },
        ),
    ):
        if hasattr(message, "result"):
            print(message.result)


asyncio.run(main())
```

**TypeScript:**

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Review the authentication module for security issues",
  options: {
    allowedTools: ["Read", "Grep", "Glob", "Agent"],
    agents: {
      "code-reviewer": {
        description: "Expert code review specialist. Use for quality, security, and maintainability reviews.",
        prompt: `You are a code review specialist with expertise in security, performance, and best practices.

When reviewing code:
- Identify security vulnerabilities
- Check for performance issues
- Verify adherence to coding standards
- Suggest specific improvements

Be thorough but concise in your feedback.`,
        tools: ["Read", "Grep", "Glob"],
        model: "sonnet"
      },
      "test-runner": {
        description: "Runs and analyzes test suites. Use for test execution and coverage analysis.",
        prompt: `You are a test execution specialist. Run tests and provide clear analysis of results.

Focus on:
- Running test commands
- Analyzing test output
- Identifying failing tests
- Suggesting fixes for failures`,
        tools: ["Bash", "Read", "Grep"]
      }
    }
  }
})) {
  if ("result" in message) console.log(message.result);
}
```

### AgentDefinition Configuration

| Field         | Type       | Required | Description                                                        |
| :------------ | :--------- | :------- | :----------------------------------------------------------------- |
| `description` | `string`   | Yes      | Natural language description of when to use this agent             |
| `prompt`      | `string`   | Yes      | The agent's system prompt defining its role and behavior           |
| `tools`       | `string[]` | No       | Array of allowed tool names. If omitted, inherits all tools        |
| `model`       | `string`   | No       | `'sonnet'`, `'opus'`, `'haiku'`, or `'inherit'`. Defaults to main |
| `skills`      | `string[]` | No       | List of skill names available to this agent                        |
| `memory`      | `string`   | No       | Memory source: `'user'`, `'project'`, or `'local'` (Python only)  |
| `mcpServers`  | `array`    | No       | MCP servers by name or inline config                               |

> **Note**: Subagents cannot spawn their own subagents. Don't include `Agent` in a subagent's `tools` array.

### What Subagents Inherit

| The subagent receives                                                          | The subagent does NOT receive                          |
| :----------------------------------------------------------------------------- | :----------------------------------------------------- |
| Its own system prompt (`AgentDefinition.prompt`) and the Agent tool's prompt   | The parent's conversation history or tool results      |
| Project CLAUDE.md (loaded via `settingSources`)                                | Skills (unless listed in `AgentDefinition.skills`)     |
| Tool definitions (inherited from parent, or the subset in `tools`)             | The parent's system prompt                             |

The parent receives the subagent's final message verbatim as the Agent tool result, but may summarize it in its own response.

### Invoking Subagents in the SDK

**Automatic invocation**: Claude decides based on the task and each subagent's `description`.

**Explicit invocation**: Mention the subagent by name:
```
"Use the code-reviewer agent to check the authentication module"
```

**Dynamic agent configuration**: Create agents dynamically based on runtime conditions:

```python
def create_security_agent(security_level: str) -> AgentDefinition:
    is_strict = security_level == "strict"
    return AgentDefinition(
        description="Security code reviewer",
        prompt=f"You are a {'strict' if is_strict else 'balanced'} security reviewer...",
        tools=["Read", "Grep", "Glob"],
        model="opus" if is_strict else "sonnet",
    )

# Usage
agents={"security-reviewer": create_security_agent("strict")}
```

### Detecting Subagent Invocation

Check for `tool_use` blocks where `name` is `"Agent"` (or `"Task"` for older SDK versions). Messages from within a subagent's context include a `parent_tool_use_id` field.

**Python:**

```python
async for message in query(prompt="...", options=options):
    if hasattr(message, "content") and message.content:
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name in ("Task", "Agent"):
                print(f"Subagent invoked: {block.input.get('subagent_type')}")

    if hasattr(message, "parent_tool_use_id") and message.parent_tool_use_id:
        print("  (running inside subagent)")
```

**TypeScript:**

```typescript
for await (const message of query({prompt: "...", options})) {
  const msg = message as any;
  for (const block of msg.message?.content ?? []) {
    if (block.type === "tool_use" && (block.name === "Task" || block.name === "Agent")) {
      console.log(`Subagent invoked: ${block.input.subagent_type}`);
    }
  }
  if (msg.parent_tool_use_id) {
    console.log("  (running inside subagent)");
  }
}
```

### Resuming Subagents (SDK)

1. **Capture the session ID**: Extract `session_id` from messages during the first query
2. **Extract the agent ID**: Parse `agentId` from the message content
3. **Resume the session**: Pass `resume: sessionId` in the second query's options

```typescript
let agentId: string | undefined;
let sessionId: string | undefined;

// First invocation
for await (const message of query({
  prompt: "Use the Explore agent to find all API endpoints",
  options: { allowedTools: ["Read", "Grep", "Glob", "Agent"] }
})) {
  if ("session_id" in message) sessionId = message.session_id;
  const extractedId = extractAgentId(message);
  if (extractedId) agentId = extractedId;
  if ("result" in message) console.log(message.result);
}

// Resume with follow-up
if (agentId && sessionId) {
  for await (const message of query({
    prompt: `Resume agent ${agentId} and list the top 3 most complex endpoints`,
    options: { allowedTools: ["Read", "Grep", "Glob", "Agent"], resume: sessionId }
  })) {
    if ("result" in message) console.log(message.result);
  }
}
```

### Tool Restrictions (SDK)

```python
agents={
    "code-analyzer": AgentDefinition(
        description="Static code analysis and architecture review",
        prompt="You are a code architecture analyst...",
        tools=["Read", "Grep", "Glob"],  # Read-only
    )
}
```

### Common Tool Combinations

| Use case           | Tools                              | Description                                    |
| :----------------- | :--------------------------------- | :--------------------------------------------- |
| Read-only analysis | `Read`, `Grep`, `Glob`            | Can examine code but not modify or execute     |
| Test execution     | `Bash`, `Read`, `Grep`            | Can run commands and analyze output            |
| Code modification  | `Read`, `Edit`, `Write`, `Grep`, `Glob` | Full read/write access without command execution |
| Full access        | All tools                          | Inherits all tools from parent (omit `tools`)  |

### Troubleshooting (SDK)

- **Claude not delegating**: Include `Agent` in `allowedTools`; mention the subagent by name; write a clear description
- **Filesystem-based agents not loading**: Agents in `.claude/agents/` are loaded at startup only. Restart the session
- **Windows long prompt failures**: Command line length limits (8191 chars). Keep prompts concise or use filesystem-based agents

---

## Agent Teams

> **Warning**: Agent teams are experimental and disabled by default. Enable via `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`. Requires Claude Code v2.1.32+.

Agent teams let you coordinate multiple Claude Code instances working together. One session acts as the team lead, coordinating work, assigning tasks, and synthesizing results. Teammates work independently, each in its own context window, and communicate directly with each other.

### When to Use Agent Teams

Best use cases:
- **Research and review**: multiple teammates investigate different aspects simultaneously
- **New modules or features**: teammates each own a separate piece
- **Debugging with competing hypotheses**: teammates test different theories in parallel
- **Cross-layer coordination**: changes spanning frontend, backend, and tests

Agent teams add coordination overhead and use significantly more tokens. For sequential tasks, same-file edits, or work with many dependencies, a single session or subagents are more effective.

### Compare with Subagents

|                   | Subagents                                        | Agent Teams                                         |
| :---------------- | :----------------------------------------------- | :-------------------------------------------------- |
| **Context**       | Own context window; results return to caller     | Own context window; fully independent               |
| **Communication** | Report results back to main agent only           | Teammates message each other directly               |
| **Coordination**  | Main agent manages all work                      | Shared task list with self-coordination             |
| **Best for**      | Focused tasks where only the result matters      | Complex work requiring discussion and collaboration |
| **Token cost**    | Lower: results summarized back to main context   | Higher: each teammate is a separate Claude instance |

### Enable Agent Teams

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

### Start Your First Agent Team

```
I'm designing a CLI tool that helps developers track TODO comments across
their codebase. Create an agent team to explore this from different angles: one
teammate on UX, one on technical architecture, one playing devil's advocate.
```

Use `Shift+Down` to cycle through teammates. After the last teammate, wraps back to the lead.

### Control Your Agent Team

#### Display Modes

- **In-process**: all teammates in main terminal. Use `Shift+Down` to cycle. Works in any terminal.
- **Split panes**: each teammate gets its own pane. Requires tmux or iTerm2.

Default is `"auto"` (split panes if inside tmux, otherwise in-process).

```json
{ "teammateMode": "in-process" }
```

```bash
claude --teammate-mode in-process
```

#### Specify Teammates and Models

```
Create a team with 4 teammates to refactor these modules in parallel.
Use Sonnet for each teammate.
```

#### Require Plan Approval

```
Spawn an architect teammate to refactor the authentication module.
Require plan approval before they make any changes.
```

When a teammate finishes planning, it sends a plan approval request to the lead. The lead reviews and either approves or rejects with feedback.

#### Talk to Teammates Directly

- **In-process mode**: `Shift+Down` to cycle, type to message. `Enter` to view, `Escape` to interrupt. `Ctrl+T` for task list.
- **Split-pane mode**: click into a teammate's pane.

#### Assign and Claim Tasks

The shared task list coordinates work. Tasks have three states: pending, in progress, and completed. Tasks can depend on other tasks.

- **Lead assigns**: tell the lead which task to give to which teammate
- **Self-claim**: after finishing, a teammate picks up the next unassigned, unblocked task

Task claiming uses file locking to prevent race conditions.

#### Shut Down Teammates

```
Ask the researcher teammate to shut down
```

#### Clean Up

```
Clean up the team
```

Always use the lead to clean up. Teammates should not run cleanup.

#### Quality Gates with Hooks

| Hook Event       | When it fires                      | Exit code 2 behavior              |
| :--------------- | :--------------------------------- | :--------------------------------- |
| `TeammateIdle`   | Teammate about to go idle          | Send feedback, keep working        |
| `TaskCreated`    | Task being created                 | Prevent creation, send feedback    |
| `TaskCompleted`  | Task being marked complete         | Prevent completion, send feedback  |

### How Agent Teams Work

#### Architecture

| Component     | Role                                                                |
| :------------ | :------------------------------------------------------------------ |
| **Team lead** | Main session that creates team, spawns teammates, coordinates work  |
| **Teammates** | Separate Claude Code instances working on assigned tasks            |
| **Task list** | Shared list of work items that teammates claim and complete         |
| **Mailbox**   | Messaging system for communication between agents                   |

Storage:
- **Team config**: `~/.claude/teams/{team-name}/config.json`
- **Task list**: `~/.claude/tasks/{team-name}/`

#### Use Subagent Definitions for Teammates

Reference a subagent type when spawning a teammate:

```
Spawn a teammate using the security-reviewer agent type to audit the auth module.
```

The teammate honors that definition's `tools` allowlist and `model`. The definition's body is appended to the teammate's system prompt as additional instructions (not replacing it). Team coordination tools (`SendMessage`, task management) are always available.

> **Note**: The `skills` and `mcpServers` frontmatter fields are NOT applied when running as a teammate.

#### Permissions

Teammates start with the lead's permission settings. After spawning, you can change individual teammate modes, but can't set per-teammate modes at spawn time.

#### Context and Communication

Each teammate has its own context window. When spawned, loads same project context as a regular session (CLAUDE.md, MCP servers, skills). The lead's conversation history does NOT carry over.

- **Automatic message delivery**: messages are delivered automatically to recipients
- **Idle notifications**: teammate notifies lead when finished
- **Shared task list**: all agents can see task status
- **message**: send to one specific teammate
- **broadcast**: send to all (use sparingly)

#### Token Usage

Token usage scales linearly with active teammates. For research, review, and new feature work, usually worthwhile. For routine tasks, a single session is more cost-effective.

### Use Case Examples

#### Parallel Code Review

```
Create an agent team to review PR #142. Spawn three reviewers:
- One focused on security implications
- One checking performance impact
- One validating test coverage
Have them each review and report findings.
```

#### Investigating with Competing Hypotheses

```
Users report the app exits after one message instead of staying connected.
Spawn 5 agent teammates to investigate different hypotheses. Have them talk to
each other to try to disprove each other's theories, like a scientific
debate. Update the findings doc with whatever consensus emerges.
```

### Best Practices (Teams)

- **Give teammates enough context**: include task-specific details in spawn prompt (they don't inherit lead's conversation history)
- **Choose appropriate team size**: start with 3-5 teammates; 5-6 tasks per teammate
- **Size tasks appropriately**: self-contained units that produce clear deliverables
- **Wait for teammates to finish**: tell the lead to wait if it starts implementing itself
- **Start with research and review**: if new to agent teams
- **Avoid file conflicts**: break work so each teammate owns different files
- **Monitor and steer**: check in on progress, redirect approaches that aren't working

### Troubleshooting (Teams)

- **Teammates not appearing**: press `Shift+Down` to cycle; check tmux is installed for split-pane mode
- **Too many permission prompts**: pre-approve common operations in permission settings
- **Teammates stopping on errors**: give additional instructions or spawn replacements
- **Lead shuts down early**: tell it to keep going or wait for teammates
- **Orphaned tmux sessions**: `tmux ls` then `tmux kill-session -t <session-name>`

### Limitations

- **No session resumption with in-process teammates**: `/resume` and `/rewind` don't restore teammates
- **Task status can lag**: teammates sometimes fail to mark tasks complete
- **Shutdown can be slow**: teammates finish current request before shutting down
- **One team per session**: clean up before starting a new one
- **No nested teams**: teammates cannot spawn their own teams
- **Lead is fixed**: can't promote a teammate or transfer leadership
- **Permissions set at spawn**: all teammates start with lead's mode
- **Split panes require tmux or iTerm2**: not supported in VS Code terminal, Windows Terminal, or Ghostty
