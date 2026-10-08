"""Where configuration comes from, and in what order.

Four layers, highest first:

1. **Constructor arguments.** What the caller passed in code.
2. **The environment.** Including ``KOJUTSU_INSTANCE`` and ``KOJUTSU_CONFIG``.
3. **The instance table** of ``kojutsu.toml``, if one is selected.
4. **The root table** of the same file.
5. **``.env``**, last as a file. Still read, still the home for credentials.

**The instance table sits above ``.env`` and that is the whole argument for the
order.** ``.env`` is untracked and per-developer; ``kojutsu.toml`` is committed
and reviewed. If ``.env`` came first, a committed file could be silently overridden
by whatever happens to be on one machine, and the file that appears in the diff would
be decorative -- the reader would be reviewing a document with no effect. The one
layer that is per-developer and untracked is the one that yields.

The environment still wins over both, because an operator overriding one value for
one run is a real thing to do and reading a file to change it would be worse.

**Credentials are refused in ``kojutsu.toml``, not merely discouraged.** That file
holds nothing secret, which is what makes it safe to commit, review, and paste into a
pull request. A file with that property only keeps it if something enforces it, and
the enforcement has to be at load time: by the time a token is in a committed file
the interesting moment has passed. Credentials live in ``.env`` and the environment,
where they are gitignored and unset by default. This sits with the other refusals
this program makes -- an over-scoped classic token, an allowlist wildcard, a
non-asserted ``capture_source`` on an evaluation -- rather than in a comment asking
politely.
"""

from __future__ import annotations

import os
import tomllib
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

#: Names that carry a credential. Present-and-non-empty in ``kojutsu.toml`` is
#: refused, so a new secret cannot be added to that file without this set being
#: updated -- and updating it is a decision somebody has to notice they are making.
SECRET_SETTING_NAMES = frozenset(
    {
        "dev_console_token",
        "github_token",
        "github_webhook_secret",
        "jira_api_token",
        "llm_api_key",
        "tanseki_api_key",
    }
)

#: Where the root settings live inside the file. An explicit table rather than bare
#: top-level keys, for one reason: a bare key that is misspelled is ignored, and a
#: mistyped ``tanseki_collections`` in a committed file would leave the store pointed
#: somewhere the author never wrote down. Naming the table makes an unrecognised
#: top-level key a refusal instead.
DEFAULTS_KEY = "defaults"

#: Where the instance tables live. A fixed key, so a settings value and an instance
#: name cannot be confused for each other: a typo in an instance name is refused
#: rather than read as a setting called ``instances``.
INSTANCES_KEY = "instances"

DEFAULT_CONFIG_FILENAME = "kojutsu.toml"

CONFIG_ENV_VAR = "KOJUTSU_CONFIG"
INSTANCE_ENV_VAR = "KOJUTSU_INSTANCE"

#: Settings the code reads as a comma-separated string, and which may therefore be
#: written as a TOML array instead. The model keeps ``str`` rather than accepting a
#: union, because an environment variable can only ever be a string: widening the
#: field would leave one spelling for the file and another for the environment, and
#: the environment is the one that cannot change. Normalising here means both spellings
#: arrive at the same ``str``.
COMMA_LIST_SETTINGS = frozenset(
    {
        "github_webhook_allowed_repositories",
        "github_webhook_repos",
        "llm_allowed_repositories",
    }
)


class ConfigError(ValueError):
    """The configuration is not usable as written.

    Distinct from pydantic's own validation error so a caller can tell "you asked for
    something that does not exist" from "the value you gave is the wrong shape", and
    report them differently.
    """


def resolve_config_path(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Find the configuration file, as an absolute path.

    Absolute because the alternative has already gone wrong: ``.env`` was resolved
    against the working directory, so a worktree with no ``.env`` of its own came up
    empty and a console started there answered ``{"error":"Tanseki is not configured."}``
    while the identical build in another checkout answered 200. Configuration that
    depends on where you happen to be standing is ambient state wearing a
    config file's clothes.

    Searched in order: an explicit path, then ``$KOJUTSU_CONFIG``, then
    ``kojutsu.toml`` in the working directory. Returns ``None`` when there is no
    file, which is a supported configuration and not a failure -- the program runs
    from the environment alone.
    """
    if explicit is not None:
        candidate = Path(explicit)
        if not candidate.is_file():
            raise ConfigError(
                f"{CONFIG_ENV_VAR} names {candidate} but there is no file there. "
                "Point it at an existing file, or unset it to run on the environment alone."
            )
        return candidate.resolve()

    raw_env = os.getenv(CONFIG_ENV_VAR)
    if raw_env is not None:
        # Set-but-empty is an opt-out, not an absence, and it means there is no file.
        # The distinction is load-bearing rather than pedantic: this file is meant to
        # be committed, so it will exist in the repository root, and the test suite
        # runs with that root as its working directory. Without an explicit way to say
        # "no file", every test would read whatever the developer last committed --
        # which fails as a test asserting against somebody else's collection rather
        # than as an error. `KOJUTSU_ENV_FILE=""` already means exactly this for
        # `.env` in this program, so the idiom is established rather than invented here.
        if not raw_env.strip():
            return None
        candidate = Path(raw_env.strip())
        if not candidate.is_file():
            raise ConfigError(
                f"{CONFIG_ENV_VAR} is set to {raw_env!r} but there is no file there. "
                "Point it at an existing file, or set it to the empty string to run on "
                "the environment alone."
            )
        return candidate.resolve()

    local = Path.cwd() / DEFAULT_CONFIG_FILENAME
    return local.resolve() if local.is_file() else None


def load_config_file(path: Path) -> dict[str, Any]:
    """Read the file once, so the credential guard and the values share a parse.

    Reading twice would let them disagree: the guard inspecting one parse and the
    settings being built from another is a way to ship a file the guard rejected and
    the loader accepted.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path} could not be read as TOML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain tables at the top level, not {type(raw).__name__}.")
    return raw


def _refuse_secrets(raw: dict[str, Any], path: Path) -> None:
    """Refuse a credential in the committed file.

    Checked at every level of the file, not only the root, because an operator who
    put a token under an instance table has still put a token in a committed file.
    An empty string is allowed: it is the shape ``.env.example`` documents, and
    refusing it would make the file that tells an operator where a credential goes
    itself unusable there.
    """
    offending: list[str] = []

    def walk(table: dict[str, Any], prefix: str) -> None:
        for key, value in table.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                walk(value, f"{name}.")
                continue
            if key in SECRET_SETTING_NAMES and str(value).strip():
                offending.append(name)

    walk(raw, "")
    if offending:
        listed = ", ".join(sorted(offending))
        raise ConfigError(
            f"{path} sets {listed}, which is a credential. This file is meant to be "
            f"committed, reviewed, and pasted into a pull request, so it must hold no "
            f"secrets: everything in it ends up somewhere a credential does not belong. "
            f"Put it in .env or the environment instead, where it is gitignored."
        )


def _normalise(table: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a TOML array into the comma-separated string the model expects.

    ``["acme/a", "acme/b"]`` is how a list is written in TOML and how an
    operator would expect to write one; ``"a,b"`` is how it has to be written in an
    environment variable. Both arrive as the same ``str`` so the allowlist parser and
    the LLM gate have exactly one shape to read.

    A non-list of strings is refused rather than coerced, because ``str(["a"])``
    yields the text ``"['a']"`` -- a value that parses as a repository named ``['a']``
    and is therefore an allowlist entry nobody wrote.
    """
    out: dict[str, Any] = {}
    for key, value in table.items():
        if key in COMMA_LIST_SETTINGS and isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise ConfigError(
                    f"{key} must be a list of strings, got "
                    f"{[type(item).__name__ for item in value]}. An entry that is not "
                    f"a string cannot be a repository name."
                )
            out[key] = ",".join(item.strip() for item in value)
            continue
        out[key] = value
    return out


def _refuse_nested_tables(table: dict[str, Any], where: str, path: Path) -> None:
    """Refuse a table inside a settings table.

    Every setting is a scalar or a list of scalars, so a nested table is always a
    mistake: a misspelled key (``llm = {model = "x"}``) or a section that reads like
    inheritance (``[instances.a.b]``). Neither is implemented, and ``extra="ignore"``
    would drop both without a sound, leaving the author believing a value was in
    effect. A section that looks like inheritance is the more dangerous of the two,
    because it looks like it is doing something.
    """
    nested = sorted(key for key, value in table.items() if isinstance(value, dict))
    if nested:
        listed = ", ".join(nested)
        raise ConfigError(
            f"{where} in {path} has nested table(s) {listed}. Settings are scalars or "
            f"lists of scalars, and a nested table is always a mistake -- a "
            f"misspelled key, or a section that reads like inheritance but is not "
            f"implemented. It is refused rather than ignored, because ignoring it "
            f"stores a configuration that does not say what the author wrote."
        )


def _root_table(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    """The ``[defaults]`` table, and a refusal for anything else at the top level.

    Every other top-level key is an error rather than a shrug. A file whose
    ``[defualts]`` is misspelled would otherwise load as an empty configuration, and
    the program would run against its built-in defaults -- pointed at
    ``kojutsu-real``, collecting from no repositories, with nothing in the output
    to distinguish that from a correct run of an empty configuration.
    """
    unrecognised = sorted(set(raw) - {DEFAULTS_KEY, INSTANCES_KEY})
    if unrecognised:
        listed = ", ".join(unrecognised)
        raise ConfigError(
            f"{path} has top-level table(s) {listed}. Only [{DEFAULTS_KEY}] and "
            f"[{INSTANCES_KEY}.<name>] are read, and an unrecognised one is refused "
            f"rather than ignored: a misspelled [{DEFAULTS_KEY}] would otherwise load "
            f"as an empty configuration and the run would look successful."
        )
    defaults = raw.get(DEFAULTS_KEY, {})
    if not isinstance(defaults, dict):
        raise ConfigError(f"[{DEFAULTS_KEY}] in {path} must be a table of settings.")
    _refuse_nested_tables(defaults, f"[{DEFAULTS_KEY}]", path)
    return _normalise(defaults)


def instance_names(raw: dict[str, Any]) -> list[str]:
    """Every instance this file defines, sorted, for an error message to offer."""
    instances = raw.get(INSTANCES_KEY)
    if not isinstance(instances, dict):
        return []
    return sorted(str(name) for name in instances)


def select_instance(raw: dict[str, Any], path: Path, requested: str | None) -> dict[str, Any]:
    """Merge the root table with the selected instance table.

    A plain dict merge, so precedence inside the file is a merge rather than a second
    parser with its own idea of nesting.

    An unknown instance name is refused. The alternative -- falling back to the root
    table -- would point capture at ``kojutsu-real`` and write a corpus nobody
    asked for, with nothing in the output to say so, which is the failure this
    feature exists to remove.
    """
    root = _root_table(raw, path)

    if not requested:
        return root

    instances = raw.get(INSTANCES_KEY)
    if not isinstance(instances, dict):
        raise ConfigError(
            f"instance {requested!r} was selected but {path} defines no "
            f"[{INSTANCES_KEY}.<name>] tables."
        )
    table = instances.get(requested)
    if not isinstance(table, dict):
        offered = ", ".join(instance_names(raw)) or "none"
        raise ConfigError(
            f"instance {requested!r} is not defined in {path}. Available: {offered}. "
            f"An unknown name is refused rather than ignored, because ignoring it "
            f"would silently run against the default collection."
        )
    _refuse_nested_tables(table, f"[{INSTANCES_KEY}.{requested}]", path)
    _refuse_secrets(table, path)
    return {**root, **_normalise(table)}


#: The file this construction resolved to, so the source and the object agree.
#: A ContextVar rather than a module global because the webhook server constructs
#: settings on more than one thread, and a shared global there would let one
#: request's instance selection decide another's -- which, for a file that chooses
#: which corpus gets written, is not a race worth having.
_CONFIG_PATH: ContextVar[Path | None] = ContextVar("kojutsu_config_path", default=None)

#: Explicit instance selection, set by the CLI ``--instance`` callback.
#: A ContextVar rather than ``os.environ`` mutation so one invocation cannot leak
#: into another thread's construction: the environment is process-wide, the
#: selection is per-invocation. ``selected_instance_name`` falls back to
#: ``KOJUTSU_INSTANCE`` so tests and operators that export the variable keep working.
_INSTANCE_OVERRIDE: ContextVar[str | None] = ContextVar("kojutsu_instance_override", default=None)


def set_selected_instance(name: str | None) -> None:
    """Select the ``[instances.<name>]`` table for subsequent constructions.

    ``None`` clears the explicit selection. This replaces the previous
    ``os.environ[KOJUTSU_INSTANCE]`` mutation in the CLI callback: the channel
    every ``Settings`` already reads is now set here, not in the process
    environment.
    """
    _INSTANCE_OVERRIDE.set(name.strip() or None if name is not None else None)
    reset_settings_cache()


def selected_instance_name() -> str:
    """The selected instance, or empty when none was selected."""
    override = _INSTANCE_OVERRIDE.get()
    if override:
        return override.strip()
    return os.getenv(INSTANCE_ENV_VAR, "").strip()


class _TomlInstanceSource(PydanticBaseSettingsSource):
    """The file layer: root table merged with one instance table.

    Positioned between the environment and ``.env`` by ``settings_customise_sources``
    below, not by anything about this class.

    Reads the path from :data:`_CONFIG_PATH` rather than resolving it again. Two
    resolutions could disagree -- the file could be written between them, or the
    environment could change -- and the credential guard would then inspect one file
    while the settings came from another.
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        path = _CONFIG_PATH.get()
        if path is None:
            return {}
        raw = load_config_file(path)
        _refuse_secrets(raw, path)
        requested = selected_instance_name() or None
        return select_instance(raw, path, requested)


class Settings(BaseSettings):
    """Application settings, from the environment and an optional instance file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def __init__(self, **values: Any) -> None:
        values.setdefault("_env_file", os.getenv("KOJUTSU_ENV_FILE", ".env") or None)
        # Absent means "discover one"; an explicit ``None`` means "there is none",
        # which is how a caller opts out without touching the environment. Those are
        # different requests and collapsing them would make the second unreachable.
        given = "_config_file" in values
        raw = values.pop("_config_file", None)
        explicit_instance = values.pop("_instance", None)
        resolved = Path(raw) if isinstance(raw, (str, os.PathLike)) else raw
        if resolved is None and not given:
            resolved = resolve_config_path()
        token = _CONFIG_PATH.set(resolved)
        instance_token = None
        if explicit_instance is not None:
            instance_token = _INSTANCE_OVERRIDE.set(str(explicit_instance).strip() or None)
        try:
            super().__init__(**values)
        finally:
            _CONFIG_PATH.reset(token)
            if instance_token is not None:
                _INSTANCE_OVERRIDE.reset(instance_token)
        # Assigned rather than passed to ``super().__init__``: ``_config_file`` is a
        # pydantic private attribute, and a private attribute handed in as a keyword
        # is accepted by the signature and then dropped without complaint -- so the
        # object reported having read no file while the values came from one.
        self._config_file = resolved

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order the layers. See the module docstring for why this order."""
        return (
            init_settings,
            env_settings,
            _TomlInstanceSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    #: The configuration file this instance resolved to, absolute, or ``None``. Not a
    #: setting -- it starts with an underscore so pydantic treats it as private and
    #: does not expect an environment variable for it -- but it is carried on the
    #: object because a caller that cannot say which file it read cannot report where
    #: a value came from, and "which file did this run use" is the first question
    #: anybody asks when two instances disagree.
    _config_file: Path | None = None

    @property
    def config_path(self) -> Path | None:
        return self._config_file

    @property
    def instance_name(self) -> str:
        """The selected instance, or empty when none was selected."""
        return selected_instance_name()

    github_token: str = ""
    github_webhook_secret: str = ""
    github_webhook_register: bool = False
    github_webhook_repos: str = ""
    github_webhook_allowed_repositories: str = ""
    github_webhook_cleanup: bool = False
    #: How many GitHub reads a `backfill-reviews` run may have in flight at once.
    #:
    #: Eight is `HISTORY_READ_CONCURRENCY` in `integrations/github.py`, spelled out
    #: rather than imported because that module imports this one, and pinned by a test
    #: so the two cannot drift. That constant is where the reasoning lives, and an
    #: operator changing the number should read it first: the bound is on *concurrency*
    #: and not on the request *rate*, so eight streams spend a fixed walk's quota in an
    #: eighth of the time.
    #:
    #: Lower this to 1 on a shared token. That is not "disable concurrency" — it is the
    #: strictly sequential walk this replaced, so it is the old behaviour exactly, with
    #: the speculation waste going to zero.
    github_history_concurrency: int = Field(default=8, ge=1, le=32)
    jira_url: str = ""
    jira_username: str = ""
    jira_api_token: str = ""
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_external_enabled: bool = False
    llm_allowed_repositories: str = ""
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300, allow_inf_nan=False)
    llm_retries: int = Field(default=1, ge=0, le=10)
    ollama_url: str = "http://localhost:11434"

    # Tanseki knowledge store (the consumer seam). Required: Tanseki is the system of
    # record for captured knowledge.
    tanseki_url: str = ""
    tanseki_api_key: str = ""
    tanseki_collection: str = "kojutsu-real"
    tanseki_timeout_seconds: float = 10.0
    tanseki_outbox_path: str = "~/.kojutsu/tanseki-outbox.db"

    # Read log: a local record of what the read MCP server was asked, and what it
    # answered. Off by default, and that default is the decision rather than a
    # placeholder: a read log is a behavioural record, it cannot be backfilled,
    # and enabling it fixes a start date that no report can move earlier.
    read_log_enabled: bool = False
    read_log_path: str = "~/.kojutsu/read-log.jsonl"
    # A week is long enough to see whether a change was informed by a read and
    # short enough that the log does not become a durable activity trail. There is
    # no setting that turns the bound off: it can be widened, not removed.
    read_log_max_age_days: int = Field(default=7, ge=1, le=365)
    # A busy day can outgrow any age bound, so the count is bounded too. Ten
    # thousand events is roughly a day of heavy use, and it keeps the retention
    # check that runs on every read in the low milliseconds.
    read_log_max_entries: int = Field(default=10_000, ge=1, le=1_000_000)

    dev_console_token: str = ""

    # Local SQLite registry for ingestion state (questions, dedupe, sessions).
    kojutsu_registry_path: str = "~/.kojutsu/registry.db"
    kojutsu_sqlite_workers: int = 1
    kojutsu_sqlite_replicas: int = 1

    # The design phase's two local stores, and the reasoning that is the reason there are
    # two of them.
    #
    # `SqlitePlanApprovalLedger` records what a person decided about a plan and
    # `SqliteTicketSink` creates the tickets that decision authorises. They are separate
    # files because a component that could both authorise and spend an authorisation could
    # authorise itself, in one transaction, leaving a record indistinguishable from one a
    # person made. That separation is the feature, so it is expressed in the file paths
    # rather than in a convention somebody can collapse while tidying.
    #
    # **Both defaults are working stores, and there is deliberately no "off" spelling.**
    # An approval ledger is the one setting where "not configured" is most tempting to
    # read as "nothing is permitted", and that reading is exactly backwards: a missing
    # ledger must not be able to mean *approvals are not enforced*, because the command
    # would then create tickets with nobody having approved anything. So unset means
    # "here is a working local store", and empty is refused rather than treated as a
    # default — see `kojutsu design` for where that refusal is enforced.
    #
    # The distinction `[instances.pilot]` exists to show, applied to a path: unset and
    # empty are deliberately different answers, and the difference is written down so the
    # next reader does not collapse it. An allowlist can afford "unset and empty read the
    # same" because an empty allowlist denies everything; an approval ledger cannot,
    # because an unusable one has to be an error and not a permission already granted.
    design_plan_approval_ledger_path: str = "~/.kojutsu/design-approvals.db"
    design_ticket_sink_path: str = "~/.kojutsu/design-tickets.db"

    @property
    def jira_base_url(self) -> str:
        """Jira base URL without trailing slash."""
        return self.jira_url.rstrip("/")

    @property
    def tanseki_enabled(self) -> bool:
        """Whether the Tanseki seam is configured."""
        return bool(self.tanseki_url.strip())


#: Repository scope mapping, in one place so the three names cannot drift.
#:
#: * ``github_webhook_allowed_repositories`` — the allowlist. Both capture and
#:   read authorise against it (see ``kojutsu.allowlist``). This is the setting
#:   of record.
#: * ``github_webhook_repos`` — legacy alias for the registration list. Read only
#:   as a fallback by :func:`webhook_registration_repos`; new configuration
#:   should set the allowlist instead.
#: * ``llm_allowed_repositories`` — a separate gate for external LLM processing,
#:   not a webhook scope. It answers "may model output leave the boundary for
#:   this repo", not "may this repo be captured". Kept distinct on purpose.
def webhook_registration_repos(settings: Settings) -> list[str]:
    """Repos to register webhooks for: allowlist first, legacy fallback.

    The lifecycle previously read only ``github_webhook_repos`` while
    authorisation read only ``github_webhook_allowed_repositories``, so a
    deployment that set one but not the other registered nowhere or authorised
    nowhere. The allowlist wins when set; the legacy value fills the gap.
    """
    primary = [
        r.strip() for r in settings.github_webhook_allowed_repositories.split(",") if r.strip()
    ]
    if primary:
        return primary
    return [r.strip() for r in settings.github_webhook_repos.split(",") if r.strip()]


def repository_scope_map(settings: Settings) -> dict[str, list[str]]:
    """Documented mapping of the three repo-scope settings to their roles."""
    return {
        "webhook_allowlist": [
            r.strip() for r in settings.github_webhook_allowed_repositories.split(",") if r.strip()
        ],
        "webhook_registration_legacy": [
            r.strip() for r in settings.github_webhook_repos.split(",") if r.strip()
        ],
        "llm_allowed": [
            r.strip() for r in settings.llm_allowed_repositories.split(",") if r.strip()
        ],
    }


_settings_lock: Any = None

_CachedSettings: Settings | None = None
_CachedEnviron: dict[str, str] | None = None
_CachedInstance: str | None = None
_CachedConfigEnv: str | None = None
_CachedEnvFile: str | None = None

try:
    import threading as _threading

    _settings_lock = _threading.Lock()
except Exception:  # pragma: no cover - threading is always available
    _settings_lock = None


def reset_settings_cache() -> None:
    """Clear the cached :func:`get_settings` instance (tests and reselection)."""
    global _CachedSettings, _CachedEnviron, _CachedInstance, _CachedConfigEnv, _CachedEnvFile
    _CachedSettings = None
    _CachedEnviron = None
    _CachedInstance = None
    _CachedConfigEnv = None
    _CachedEnvFile = None


def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, building it once.

    Rebuilt when the observable inputs change (environment snapshot, selected
    instance, ``KOJUTSU_CONFIG``/``KOJUTSU_ENV_FILE`` pointers) so tests that
    mutate the environment via ``monkeypatch`` still see fresh values while
    steady-state production code constructs exactly once per process.
    """
    global _CachedSettings, _CachedEnviron, _CachedInstance, _CachedConfigEnv, _CachedEnvFile
    lock = _settings_lock
    if lock is not None:
        lock.acquire()
    try:
        snapshot = dict(os.environ)
        instance = selected_instance_name()
        config_env = os.getenv(CONFIG_ENV_VAR)
        env_file = os.getenv("KOJUTSU_ENV_FILE")
        if (
            _CachedSettings is not None
            and _CachedEnviron == snapshot
            and _CachedInstance == instance
            and _CachedConfigEnv == config_env
            and _CachedEnvFile == env_file
        ):
            return _CachedSettings
        settings = Settings()
        _CachedSettings = settings
        _CachedEnviron = snapshot
        _CachedInstance = instance
        _CachedConfigEnv = config_env
        _CachedEnvFile = env_file
        return settings
    finally:
        if lock is not None:
            lock.release()
