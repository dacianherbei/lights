# Claude Code Configuration - SPARC Development Environment

## 🚨 CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently, not just MCP

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### 🎯 CRITICAL: Claude Code Task Tool for Agent Execution

**Claude Code's Task tool is the PRIMARY way to spawn agents:**
```javascript
// ✅ CORRECT: Use Claude Code's Task tool for parallel agent execution
[Single Message]:
  Task("Research agent", "Analyze requirements and patterns...", "researcher")
  Task("Coder agent", "Implement core features...", "coder")
  Task("Tester agent", "Create comprehensive tests...", "tester")
  Task("Reviewer agent", "Review code quality...", "reviewer")
  Task("Architect agent", "Design system architecture...", "system-architect")
```

### 📁 File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code

# Fulgor IDE - LLVM Cross-Language Bridge

## Project Overview

Fulgor is a revolutionary node-based visual programming IDE with zero-overhead cross-language execution via LLVM. The project consists of three main crates plus supporting components.

## Repository Structure

```
LLVM_cross_language_bridge/
├── crates/
│   ├── forge/          # Compilation engine (LLVM, bridge generation, cross-lang types)
│   ├── nexus/          # Execution engine (orchestration, pods, strategies, IPC)
│   ├── metadata/       # Shared types between forge and nexus (no circular deps)
│   ├── optical/        # UI component library (eframe/egui on wgpu today; full GUI library on lights)
│   ├── atrium/         # Application coordinator (project model, compilation, execution)
│   └── fulgor/         # Workspace re-export crate
├── auto_bridge_generator/  # Automatic FFI bridge code generation
├── running_connections/    # End-to-end pipeline experiments (forge→nexus)
├── experiments/            # Research experiments (B.4-B.8 validation)
├── exercises/              # Learning exercises (week1-week2)
├── docs/                   # Architecture docs, ADRs, reflections
└── Cargo.toml              # Workspace root
```

## Build & Test Commands

```bash
# Build entire workspace
cargo build

# Build specific crate
cargo build --package nexus
cargo build --package forge
cargo build --package metadata

# Run all tests
cargo test

# Run tests for specific crate
cargo test --package nexus
cargo test --package forge

# lights: use the script, NOT plain `cargo test`. `fit` is not a default feature,
# so `cargo test` silently skips 157 of 408 tests — the entire fit arm, which is
# where nearly every published number comes from. The script runs --all-features
# and sums the per-target result lines itself.
bash crates/lights/scripts/run_tests.sh

# Run specific test module
cargo test --package nexus strategy::factory
cargo test --package nexus tier

# Run experiments
cargo run --bin exp00_single_node
cargo run --bin exp01_two_nodes

# Run benchmarks (forge)
cargo bench --package forge

# LLVM requirement: LLVM 21.0 must be installed
# On macOS: brew install llvm@21
# Set LLVM_SYS_210_PREFIX accordingly
```

## Architecture Overview

### Crate Responsibilities

**forge** (Compilation):
- LLVM IR generation and JIT compilation
- Cross-language type system (Rust, C, C++, COBOL, Python, Julia)
- Bridge code generation for cross-language FFI (62x better than targets)
- Memory pool system (3.5x-6.9x speedup over malloc)
- Binary compilation and loading

**nexus** (Execution):
- Workflow orchestration with three production tiers + debug tier
- Pod lifecycle management (14-state machine)
- Strategy system: Execution, Memory, Input, Output, Logging, RuntimeOptimization
- IPC via Unix domain sockets with bincode serialization
- Process-based isolation (TierBuilder metamorphosis pattern)

**metadata** (Shared):
- Binary metadata format for forge→nexus interface
- Canonical type definitions shared between crates
- Function signatures and parameter descriptions

### Three-Tier Architecture (nexus)

| Tier | Name | Isolation | Overhead | Use Case |
|------|------|-----------|----------|----------|
| 0 | Inline | Rust only | Zero | Testing, embedded |
| 1 | Isolated | Process + threads | Low (~10μs) | Production |
| 2 | Resilient | Process + process | Higher (~100μs) | Mission-critical |
| Debug | Debug | Separate tier | Very high | Interactive debugging |

### Key Design Patterns

- **Builder Metamorphosis**: TierBuilder process transforms into Pod process (same PID)
- **Dual-Queue Priority**: Control commands (unbounded) have priority over work commands (bounded)
- **Strategy Hot-Swap**: Strategies can be replaced at runtime with state migration via bincode
- **Cooperative Shutdown**: Atomic flags for graceful shutdown (<24ms validated)

### Communication Architecture

- **Local IPC**: Unix domain sockets with length-prefixed framing + bincode v2
- **Message Protocol**: `PodCommand` / `PodNotification` / `ControlCommand` / `WorkCommand`
- **NexusManagement**: Assigns unique IDs (PodId, TierStrategyId) via atomic counters

## Development Principles

1. **Small controlled steps** - iterative, not broad changes
2. **Research before implementation** - 4h research prevents 2-4 weeks wrong work
3. **Architecture before code** - design documents precede implementation
4. **Empirical validation** - "always measure, never assume" (95% CI benchmarks)
5. **Test-driven** - tests built before implementation
6. **Configuration over heuristics** - explicit choices, no hidden complexity

## Current State (as of last update)

- B.10 (Full Local Strategy Interaction): ~75% complete
  - InlineOrchestrator: production-ready
  - IsolatedOrchestrator: production-ready
  - ResilientOrchestrator: needs completion
  - IPC infrastructure: complete (channel, transport, messages)
  - TierBuilder: implemented with process spawning
  - Strategy factory system: implemented
  - 406+ tests passing (nexus lib alone, verified May 2026)
- B.11 (Distributed Execution): not started
- B.9 (Debugging Support): not started
- B.12 (Benchmarking): not started

## Sequence: B.10 → B.11 → B.9 → B.12

Per ADR-005:
- B.10 first: "the rock on which everything else is built"
- B.11 before B.9: remote debugging requires remote executors
- B.12 last: benchmark complete system, not fragments

## Key ADRs

- **ADR-001**: Binary deployment via dlopen
- **ADR-002**: Compositional executor (strategy pattern)
- **ADR-003**: Four strategy traits → unified Strategy trait evolution
- **ADR-004**: Deferred benchmarking until complete system
- **ADR-005**: Reordered: B.10 → MVP GUI → B.11 → B.9 → B.12

## MVP GUI Phase (Next after B.10.R3)

Self-hosting validation: fulgor IDE managing its own source code.
**Crates:** `optical` (eframe/egui on wgpu today; full GUI library on lights), `atrium` (coordinator)
**4 panels:** Node graph editor, Text editor, Runtime pod view, Info/errors
**Validation:** Parse fulgor's ~125+ public Rust items as nodes, compile via forge,
execute via nexus, display pod states — full loop.
**Timeline:** 14-18 working days (~3 weeks)
See `docs/MVP_GUI_PLAN.md` for full plan.

## B.11 Distributed Architecture (Designed)

Each machine runs a **NexusService** daemon (Erlang OTP model):
- Coordinator NexusService starts the user application
- Remote NexusService is asked to compile (via local Forge) and create pods
- I/O strategies handle cross-machine data flow (location-transparent to binaries)
- Tier strategy is respected on remote machines (isolation from NexusManagement)
- Control plane: coordinator ↔ delegates via ServiceCommand/ServiceResponse
- Data plane: pod ↔ pod via NetworkInput/NetworkOutput strategies

See `docs/B11_DISTRIBUTED_ARCHITECTURE.md` for full design.

## NexusService Vision (Long-Term)

The Forge/Nexus binary + metadata separation enables use cases beyond development:
- **Closed-source distribution**: vendor ships binary + metadata, no Forge needed
- **Third-party module library**: content-addressed binary cache, blockchain monetization
- **Embedded devices**: NexusService as minimal bootloader, remote Forge cross-compiles
- **CI/CD**: NexusService IS the runner, Forge IS the build system, tiers ARE isolation
- **PGO feedback loop**: Forge embeds instrumentation → NexusService collects data →
  Forge returns DeploymentPlan (binary swap, topology change, hardware migration)
- **OS-level**: NexusService as process manager, per-user Forge-compiled binaries,
  security baked into code at compile time (no runtime permission checks)

The **version update protocol** (DeploymentPlan) evolves through versions:
v1=BinaryUpdate, v2=TopologyChange, v3=ResourceRequirements, v4=PerUserCompilation

See `docs/NEXUS_SERVICE_VISION.md` for full analysis.

## Code Style

- Rust 2024 edition
- Use `thiserror` for error types in libraries
- Use `anyhow` for error handling in binaries/tests
- Bincode v2 for serialization (requires `Encode`/`Decode<()>` traits)
- Serde for JSON/TOML configuration files
- Comprehensive doc comments on public APIs
- Test modules co-located or in `/tests/` directory

## Hardware Context

Development on Apple M5 Max (arm64, 128 GB unified memory, macOS 15).
Performance benchmarks are platform-specific - always note hardware in results,
and treat any figure measured before Aug 2026 as belonging to the previous
machine (2014 MacBook Pro, Intel Haswell i7, 16GB) rather than this one.

Cargo builds go to a RAM disk: `build.target-dir` is set globally in
`~/.cargo/config.toml` to `/Volumes/RamBuild/cargo-target`. A build landing
there is normal. `cargo clean` cleans the RAM disk, not any on-disk `target/`.

## Periodic Maintenance

### Daily sublimation

`scripts/hook_sublimation.sh` is no longer wired to fire per-commit (the entry
was removed from `.claude/settings.json`'s PostToolUse hooks because it was
producing duplicate-ID auto-spam in `docs/open_questions.toml` — 590 entries
collided historically due to a bash-octal bug in `next_qid` that has since
been fixed via `10#$max_id`).

Once per Claude session, check the most recent `title = "Sublimation:` entry's
`discovered` field in `docs/open_questions.toml`. If it's >24h old, run:

```bash
echo '{"tool_input":{"command":"git commit"}}' | bash scripts/hook_sublimation.sh
```

Then read the newly-appended entry. If its findings are textually identical to
the previous sublimation entry, **delete the new one** (no new signal — saves
space). Otherwise, keep it as a fresh tracking record. The cleaned baseline is
~60 entries; never let it grow unbounded again.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.
