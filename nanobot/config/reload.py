"""Hot-reload a running ``Config`` object in place from a freshly parsed one.

The gateway's file watcher used to invalidate only the model runtime, so every other
consumer kept whatever the boot ``Config`` held: editing ``config.json`` was live for
providers/models but not for anything else. This applies a fresh parse onto the
*existing* object graph instead of swapping pointers, because many holders keep a
reference to a sub-model rather than the root -- ``GatewayHTTPHandler.config`` is a
``WebSocketConfig`` instance, ``AgentLoop.tools_config`` is a ``ToolsConfig`` -- so
replacing ``config.tools`` would leave them reading the old object.

Shapes that actually occur in this schema are all three of: static pydantic fields
(``tools.exec.max_runtime``), extras from ``extra="allow"`` sections (``providers.*``
custom entries, and every ``channels.*`` section, which arrives as a plain dict), and
dict-of-model fields (``model_presets``).

Fields that configure construction-time resources or security boundaries are refused
instead of mutated: they genuinely need a restart, and silently rewriting them under a
running process would let one file write quietly widen the sandbox.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

#: Whole subtrees that are wired up at startup (channel connections read these).
FROZEN_PREFIXES: tuple[str, ...] = (
    "channels.",
    "agents.defaults.workspace",
)

#: Individual leaves that either need a restart or must not move at runtime.
FROZEN_PATHS: frozenset[str] = frozenset({
    # Bound sockets / listeners.
    "gateway.host",
    "gateway.port",
    "api.host",
    "api.port",
    # Security boundaries: hot-changing these is self-weakening.
    "tools.restrict_to_workspace",
    "tools.ssrf_whitelist",
    "tools.my.allow_set",
    "tools.exec.sandbox",
    "tools.exec.sandbox_ro_binds",
    "tools.exec.sandbox_rw_binds",
    "tools.exec.allowed_env_keys",
    "tools.exec.allow_patterns",
    "tools.exec.deny_patterns",
    "tools.exec.max_runtime_limit",
    # 执行超时/空闲策略在构造期拷进工具（shell.py:174-176），改了要重启才生效
    "tools.exec.max_runtime",
    "tools.exec.idle_timeout",
    "tools.exec.path_prepend",
    "tools.exec.path_append",
    # 出站路由：provider 侧会随快照重建，但 web 工具与通道在构造期就固化了 proxy，
    # 半生效比不生效更糟 —— 统一按需要重启处理
    "network.proxy",
    # API server 的 host/port/timeout 都在启动时读取（cli/commands.py:361）
    "api.timeout",
    # web 工具的 proxy/user_agent 是 create() 时算好的（tools/web.py:408-411）
    "tools.web.use_proxy",
    "tools.web.user_agent",
})

#: Registration-time switches (`tools.exec.enable`, `tools.my.enable`, ...).
FROZEN_SUFFIXES: tuple[str, ...] = (".enable",)


@dataclass(frozen=True, slots=True)
class ConfigReloadResult:
    """What a hot reload changed, what it refused, and what failed to apply."""

    applied: tuple[str, ...] = ()
    refused: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    @property
    def needs_restart(self) -> bool:
        return bool(self.refused)

    def merge(self, other: "ConfigReloadResult") -> "ConfigReloadResult":
        return ConfigReloadResult(
            self.applied + other.applied,
            self.refused + other.refused,
            self.errors + other.errors,
        )

    def summary(self) -> str:
        parts: list[str] = []
        if self.applied:
            parts.append(f"applied: {', '.join(self.applied)}")
        if self.refused:
            parts.append(f"needs restart: {', '.join(self.refused)}")
        if self.errors:
            parts.append(f"errors: {', '.join(self.errors)}")
        return "; ".join(parts) if parts else "no changes"


def is_frozen(path: str) -> bool:
    """Return True when ``path`` must not be mutated on a running process."""
    if path in FROZEN_PATHS:
        return True
    if any(path.startswith(prefix) for prefix in FROZEN_PREFIXES):
        return True
    return any(path.endswith(suffix) for suffix in FROZEN_SUFFIXES)


@dataclass
class _Tally:
    applied: list[str]
    refused: list[str]
    errors: list[str]

    def result(self) -> ConfigReloadResult:
        return ConfigReloadResult(tuple(self.applied), tuple(self.refused), tuple(self.errors))


def apply_config_in_place(
    target: BaseModel,
    fresh: BaseModel,
    *,
    prefix: str = "",
) -> ConfigReloadResult:
    """Mutate ``target`` so it matches ``fresh`` without replacing nested models.

    ``target`` keeps its identity (and so do its sub-models and dict containers), which
    is what makes holders of sub-model references observe the new values on their next
    read. Frozen paths are reported, never written.
    """
    tally = _Tally([], [], [])
    _walk_model(target, fresh, prefix, tally)
    return tally.result()


def _walk_model(target: BaseModel, fresh: BaseModel, prefix: str, tally: _Tally) -> None:
    for name in type(target).model_fields:
        path = f"{prefix}{name}"
        old_value = getattr(target, name, None)
        new_value = getattr(fresh, name, None)
        if old_value == new_value:
            continue
        _assign(target, name, path, old_value, new_value, tally)

    # extra="allow" sections: channels arrive here as plain dicts, custom providers too.
    old_extra = getattr(target, "__pydantic_extra__", None) or {}
    new_extra = getattr(fresh, "__pydantic_extra__", None) or {}
    for key in list(new_extra):
        path = f"{prefix}{key}"
        new_value = new_extra[key]
        if key not in old_extra:
            if is_frozen(path):
                tally.refused.append(path)
                continue
            old_extra[key] = new_value
            tally.applied.append(path)
            continue
        old_value = old_extra[key]
        if old_value == new_value:
            continue
        if is_frozen(path):
            tally.refused.append(path)
            continue
        _assign_extra(old_extra, key, path, old_value, new_value, tally)
    for key in [k for k in old_extra if k not in new_extra]:
        path = f"{prefix}{key}"
        if is_frozen(path):
            tally.refused.append(path)
            continue
        del old_extra[key]
        tally.applied.append(f"{path} (removed)")


def _assign(
    owner: BaseModel,
    name: str,
    path: str,
    old_value: Any,
    new_value: Any,
    tally: _Tally,
) -> None:
    if is_frozen(path):
        tally.refused.append(path)
        return
    if isinstance(old_value, BaseModel) and isinstance(new_value, BaseModel) and type(old_value) is type(new_value):
        _walk_model(old_value, new_value, f"{path}.", tally)
        return
    if isinstance(old_value, dict) and isinstance(new_value, Mapping):
        _apply_mapping(old_value, new_value, path, tally)
        return
    try:
        setattr(owner, name, new_value)
    except Exception as exc:  # one bad field must not abort the whole reload
        tally.errors.append(f"{path} ({type(exc).__name__})")
        return
    tally.applied.append(path)


def _assign_extra(
    container: dict[str, Any],
    key: str,
    path: str,
    old_value: Any,
    new_value: Any,
    tally: _Tally,
) -> None:
    if isinstance(old_value, BaseModel) and isinstance(new_value, BaseModel) and type(old_value) is type(new_value):
        _walk_model(old_value, new_value, f"{path}.", tally)
        return
    if isinstance(old_value, dict) and isinstance(new_value, Mapping):
        _apply_mapping(old_value, new_value, path, tally)
        return
    container[key] = new_value
    tally.applied.append(path)


def _apply_mapping(old: dict[str, Any], new: Mapping[str, Any], path: str, tally: _Tally) -> None:
    """Update a dict container in place so holders of that same dict see the change."""
    for key, new_value in new.items():
        child_path = f"{path}.{key}"
        if is_frozen(child_path):
            tally.refused.append(child_path)
            continue
        if key not in old:
            old[key] = new_value
            tally.applied.append(child_path)
            continue
        old_value = old[key]
        if old_value == new_value:
            continue
        if isinstance(old_value, BaseModel) and isinstance(new_value, BaseModel) and type(old_value) is type(new_value):
            _walk_model(old_value, new_value, f"{child_path}.", tally)
        elif isinstance(old_value, dict) and isinstance(new_value, Mapping):
            _apply_mapping(old_value, new_value, child_path, tally)
        else:
            old[key] = new_value
            tally.applied.append(child_path)
    for key in [k for k in old if k not in new]:
        child_path = f"{path}.{key}"
        if is_frozen(child_path):
            tally.refused.append(child_path)
            continue
        del old[key]
        tally.applied.append(f"{child_path} (removed)")
