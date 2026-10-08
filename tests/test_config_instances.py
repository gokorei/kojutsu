"""Configuration layering, and the two refusals that protect it.

Three things are worth stating before the individual tests, because they are what
the rest of this file is checking.

**The instance table outranks `.env`.** `.env` is untracked and per-developer;
`kojutsu.toml` is committed and reviewed. If `.env` came first, the file that
appears in a diff could be overridden by whatever happened to be on one machine, and
the reviewed document would have no effect. A test that only checks "the value is
right" passes under either order, so each layer is asserted against all four
competing ones.

**A credential in the committed file is refused, not discouraged.** That file is
meant to be committed, reviewed, and pasted into a pull request; those three things
are only safe while it holds nothing secret. Enforcement at load time is the only
moment it is worth anything.

**An unknown instance name is refused rather than ignored.** Ignoring it runs the
command against the default collection instead, which is the exact silent-wrong-corpus
failure this feature exists to remove.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from kojutsu.allowlist import configured_repositories
from kojutsu.config import (
    CONFIG_ENV_VAR,
    INSTANCE_ENV_VAR,
    SECRET_SETTING_NAMES,
    ConfigError,
    Settings,
    instance_names,
    resolve_config_path,
    select_instance,
)

TOML = """\
[defaults]
tanseki_collection = "kojutsu-root"
llm_model = "root/model"

[instances.secondary]
tanseki_collection = "kojutsu-secondary"
github_webhook_allowed_repositories = "acme/service,acme/library"

[instances.empty]
"""


def write(tmp_path: Path, body: str, name: str = "kojutsu.toml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def load(path: Path, instance: str | None = None) -> Settings:
    """Load ``path`` the way a caller would, without touching the environment."""
    if instance is None:
        monkey = os.environ.pop(INSTANCE_ENV_VAR, None)
        try:
            return Settings(_config_file=path)
        finally:
            if monkey is not None:
                os.environ[INSTANCE_ENV_VAR] = monkey
    previous = os.environ.get(INSTANCE_ENV_VAR)
    os.environ[INSTANCE_ENV_VAR] = instance
    try:
        return Settings(_config_file=path)
    finally:
        if previous is None:
            os.environ.pop(INSTANCE_ENV_VAR, None)
        else:
            os.environ[INSTANCE_ENV_VAR] = previous


# --- the four layers ---------------------------------------------------------


def test_an_instance_table_supplies_the_collection(tmp_path: Path) -> None:
    settings = load(write(tmp_path, TOML), "secondary")

    assert settings.tanseki_collection == "kojutsu-secondary"
    assert settings.github_webhook_allowed_repositories == ("acme/service,acme/library")


def test_the_root_table_is_used_when_no_instance_is_selected(tmp_path: Path) -> None:
    """No selection is not an error and not an empty result: it is the root table."""
    settings = load(write(tmp_path, TOML))

    assert settings.tanseki_collection == "kojutsu-root"
    assert settings.llm_model == "root/model"


def test_an_instance_inherits_the_root_table(tmp_path: Path) -> None:
    """A merge, not a replacement.

    An instance that had to restate everything would duplicate the root table per
    instance, and the two copies would drift the first time one was edited.
    """
    settings = load(write(tmp_path, TOML), "secondary")

    assert settings.tanseki_collection == "kojutsu-secondary", "the instance wins"
    assert settings.llm_model == "root/model", "the root still fills the rest"


def test_the_environment_outranks_both_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator overriding one value for one run is a real thing to do."""
    monkeypatch.setenv("TANSEKI_COLLECTION", "from-environment")

    settings = load(write(tmp_path, TOML), "secondary")

    assert settings.tanseki_collection == "from-environment"


def test_the_file_outranks_the_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The order the whole feature rests on, asserted against the layer it beats.

    `kojutsu.toml` is committed and reviewed; `.env` is neither. If `.env` won,
    the file in the diff would be a document with no effect on the machine that
    matters, and reviewing it would be theatre.
    """
    dotenv = write(tmp_path, "TANSEKI_COLLECTION=from-dotenv\n", name=".env")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(write(tmp_path, TOML)))
    monkeypatch.setenv("KOJUTSU_ENV_FILE", str(dotenv))
    monkeypatch.setenv(INSTANCE_ENV_VAR, "secondary")

    settings = Settings()

    assert settings.tanseki_collection == "kojutsu-secondary"


def test_the_dotenv_still_fills_what_the_file_is_silent_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Layered, not overridden: the file wins where it speaks and not where it does not."""
    dotenv = write(tmp_path, "LLM_PROVIDER=ollama\nTANSEKI_COLLECTION=from-dotenv\n", name=".env")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(write(tmp_path, TOML)))
    monkeypatch.setenv("KOJUTSU_ENV_FILE", str(dotenv))
    monkeypatch.setenv(INSTANCE_ENV_VAR, "secondary")

    settings = Settings()

    assert settings.tanseki_collection == "kojutsu-secondary"
    assert settings.llm_provider == "ollama"


def test_constructor_arguments_outrank_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(write(tmp_path, TOML)))
    monkeypatch.setenv(INSTANCE_ENV_VAR, "secondary")

    settings = Settings(tanseki_collection="from-code")

    assert settings.tanseki_collection == "from-code"


def test_no_file_at_all_is_a_supported_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The program must still run from the environment alone, with nothing installed."""
    monkeypatch.setenv(CONFIG_ENV_VAR, "")
    monkeypatch.setenv("TANSEKI_COLLECTION", "from-environment")
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.config_path is None
    assert settings.tanseki_collection == "from-environment"


# --- named instances ---------------------------------------------------------


def test_an_instance_can_be_named_by_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(write(tmp_path, TOML)))
    monkeypatch.setenv(INSTANCE_ENV_VAR, "secondary")

    assert Settings().tanseki_collection == "kojutsu-secondary"


def test_an_unknown_instance_is_refused_with_the_names_on_offer(tmp_path: Path) -> None:
    """Falling back to defaults would write the wrong corpus and say nothing.

    A reader seeing `kojutsu-real` in the output would have no way to tell that
    `--instance secondary` was misspelled, so the run looks successful and the records
    land in the wrong place.
    """
    path = write(tmp_path, TOML)

    with pytest.raises(ConfigError) as raised:
        load(path, "secondar")

    message = str(raised.value)
    assert "secondar" in message
    assert "secondary" in message and "empty" in message, "the available names must be listed"
    assert "kojutsu-real" not in message, "must not hint at the default it refused to use"


def test_selecting_an_instance_in_a_file_that_defines_none(tmp_path: Path) -> None:
    path = write(tmp_path, '[defaults]\ntanseki_collection = "only"\n')

    with pytest.raises(ConfigError, match="defines no"):
        load(path, "secondary")


def test_an_instance_may_be_empty(tmp_path: Path) -> None:
    """A table with nothing in it is a valid thing to write, and means "the root table"."""
    settings = load(write(tmp_path, TOML), "empty")

    assert settings.tanseki_collection == "kojutsu-root"


def test_instance_names_are_sorted_and_exclude_the_root_table(tmp_path: Path) -> None:
    path = write(tmp_path, TOML)

    assert instance_names(load_config(path)) == ["empty", "secondary"]


def load_config(path: Path) -> dict[str, Any]:
    from kojutsu.config import load_config_file

    return load_config_file(path)


def test_the_resolved_file_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller that cannot say which file it read cannot report where a value came from."""
    path = write(tmp_path, TOML)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))

    resolved = Settings().config_path

    assert resolved == path.resolve()
    assert resolved is not None and resolved.is_absolute()


def test_the_selected_instance_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(write(tmp_path, TOML)))
    monkeypatch.setenv(INSTANCE_ENV_VAR, "secondary")

    assert Settings().instance_name == "secondary"


# --- credentials in the committed file ---------------------------------------


@pytest.mark.parametrize("key", sorted(SECRET_SETTING_NAMES))
def test_a_credential_in_the_file_is_refused(tmp_path: Path, key: str) -> None:
    """One case per secret, so adding one later forces a decision rather than a gap.

    A committed file that can hold a token is a committed file that will eventually
    hold one, and the moment that matters is before it is pushed.
    """
    path = write(tmp_path, f'[defaults]\n{key} = "a-real-looking-value"\n')

    with pytest.raises(ConfigError) as raised:
        load(path)

    assert key in str(raised.value)
    assert "gitignored" in str(raised.value), "the message must say where it belongs"


def test_a_credential_under_an_instance_is_refused_too(tmp_path: Path) -> None:
    """Checked at every level, because a token under an instance is still in the file."""
    path = write(tmp_path, '[instances.secondary]\ngithub_token = "ghp_example"\n')

    with pytest.raises(ConfigError, match="github_token"):
        load(path, "secondary")


def test_an_empty_credential_is_allowed(tmp_path: Path) -> None:
    """It is the shape `.env.example` documents, and refusing it would forbid the pointer."""
    settings = load(write(tmp_path, '[defaults]\ngithub_token = ""\n'))

    assert settings.github_token == ""


def test_every_secret_the_settings_model_knows_is_guarded() -> None:
    """The guard and the model must not drift apart.

    A new secret field added to `Settings` and forgotten here would be the one
    credential the file accepts, which is the worst possible gap: it would be the
    one nobody suspects.
    """
    undeclared = {
        name
        for name in Settings.model_fields
        if any(word in name for word in ("token", "secret", "api_key"))
    } - SECRET_SETTING_NAMES

    assert undeclared == set(), f"unguarded credential fields in Settings: {sorted(undeclared)}"


# --- path resolution ---------------------------------------------------------


def test_an_explicit_path_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere.toml"

    with pytest.raises(ConfigError, match="no file there"):
        resolve_config_path(missing)


def test_a_config_env_var_naming_nothing_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently ignoring it would leave the run on defaults, which is a different program."""
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "absent.toml"))

    with pytest.raises(ConfigError, match="no file there"):
        resolve_config_path()


def test_an_empty_config_env_var_means_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opt-out the test suite depends on, and it must beat the working directory."""
    monkeypatch.chdir(tmp_path)
    write(tmp_path, TOML)
    monkeypatch.setenv(CONFIG_ENV_VAR, "")

    assert resolve_config_path() is None


def test_a_file_in_the_working_directory_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write(tmp_path, TOML)
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

    found = resolve_config_path()

    assert found is not None and found.name == "kojutsu.toml"
    assert found.is_absolute(), "a relative path is the cwd-dependence this replaced"


def test_a_malformed_file_names_itself_and_the_problem(tmp_path: Path) -> None:
    """Swallowed into defaults, a typo in a committed file looks like an unset setting."""
    path = write(tmp_path, "[instances.secondary\ntanseki_collection = 'x'\n")

    with pytest.raises(ConfigError) as raised:
        load(path)

    message = str(raised.value)
    assert str(path) in message
    assert "TOML" in message


# --- the merge itself --------------------------------------------------------


def test_the_root_table_never_leaks_the_instances_key(tmp_path: Path) -> None:
    """`instances` is a namespace, not a setting, and must not read as one."""
    merged = select_instance(load_config(write(tmp_path, TOML)), write(tmp_path, TOML), None)

    assert "instances" not in merged


def test_a_misspelled_defaults_table_is_refused(tmp_path: Path) -> None:
    """The failure this shape exists to prevent.

    `[defualts]` loads as a file with no settings in it, the program runs on its
    built-in defaults, and the run is indistinguishable from a correct one: it
    collects nothing, from nowhere, and says nothing about why.
    """
    path = write(tmp_path, '[defualts]\ntanseki_collection = "kojutsu-secondary"\n')

    with pytest.raises(ConfigError) as raised:
        load(path)

    assert "defualts" in str(raised.value)
    assert "defaults" in str(raised.value)


def test_a_file_with_only_an_instance_table_is_valid(tmp_path: Path) -> None:
    """The root table is optional; an instance may be the whole file."""
    settings = load(
        write(tmp_path, '[instances.secondary]\ntanseki_collection = "kojutsu-secondary"\n'),
        "secondary",
    )

    assert settings.tanseki_collection == "kojutsu-secondary"
    assert settings.llm_provider == "openai", "an absent root falls back to the built-in default"


def test_an_empty_file_is_valid_and_changes_nothing(tmp_path: Path) -> None:
    settings = load(write(tmp_path, ""))

    assert settings.tanseki_collection == "kojutsu-real"


def test_instance_tables_may_not_nest_further(tmp_path: Path) -> None:
    """Refused rather than ignored, on the same reasoning as a misspelled table.

    A nested `[instances.a.b]` reads like inheritance. It is not implemented, so
    honouring it would mean the deeper table's values were silently dropped and the
    author would believe an instance inherited from another.
    """
    path = write(
        tmp_path, '[instances.a]\ntanseki_collection = "x"\n[instances.a.b]\nllm_model = "y"\n'
    )

    with pytest.raises(ConfigError):
        load(path, "a")


def test_an_allowlist_may_be_a_toml_array(tmp_path: Path) -> None:
    """The idiomatic TOML spelling, normalised to the string the model already reads.

    An environment variable can only be a string, so the field stays ``str`` and the
    file layer rewrites a list. Normalising here rather than widening the field means
    one shape reaches the allowlist parser whichever layer supplied it -- a second
    spelling to read is a second thing to get wrong.
    """
    path = write(
        tmp_path,
        '[instances.secondary]\ngithub_webhook_allowed_repositories = ["acme/a", "acme/b"]\n',
    )

    settings = load(path, "secondary")

    assert settings.github_webhook_allowed_repositories == "acme/a,acme/b"
    assert sorted(configured_repositories(settings)) == ["acme/a", "acme/b"]


def test_every_comma_list_setting_accepts_an_array(tmp_path: Path) -> None:
    """One case per setting, so a new list-shaped field forces the question again."""
    from kojutsu.config import COMMA_LIST_SETTINGS

    for key in sorted(COMMA_LIST_SETTINGS):
        path = write(tmp_path, f'[defaults]\n{key} = ["one", "two"]\n')

        assert getattr(load(path), key) == "one,two", key


def test_an_allowlist_of_non_strings_is_refused(tmp_path: Path) -> None:
    """`str([...])` would produce the text "['a']" and an entry nobody wrote."""
    path = write(tmp_path, '[defaults]\nllm_allowed_repositories = ["ok", 7]\n')

    with pytest.raises(ConfigError, match="list of strings"):
        load(path)
