import datetime

import pytest

from gpt_assistant_lib._history import Role, SimpleHistory


def test_init_system_content():
    history = SimpleHistory(max_size=5, ttl=600)
    history.init_system_content("System initialized")
    entries = history.get()
    assert entries == [
        {"role": "system", "content": "System initialized"},
    ]


def test_get_entries():
    history = SimpleHistory(max_size=3, ttl=600)
    history.init_system_content("System initialized")
    history.insert(Role.USER, "User action 1")
    history.insert(Role.ASSISTANT, "Assistant action")
    history.insert(Role.USER, "User action 2")
    entries = history.get()
    assert entries == [
        {"role": "system", "content": "System initialized"},
        {"role": "user", "content": "User action 1"},
        {"role": "assistant", "content": "Assistant action"},
        {"role": "user", "content": "User action 2"},
    ]


def test_get_entries_no_system():
    history = SimpleHistory(max_size=3, ttl=600)
    history.insert(Role.USER, "User action 1")
    history.insert(Role.ASSISTANT, "Assistant action")
    history.insert(Role.USER, "User action 2")
    entries = history.get()
    assert entries == [
        {"role": "user", "content": "User action 1"},
        {"role": "assistant", "content": "Assistant action"},
        {"role": "user", "content": "User action 2"},
    ]


def test_try_insert_system_role():
    history = SimpleHistory(max_size=5, ttl=600)
    with pytest.raises(ValueError, match="Can't insert system role entry"):
        history.insert(Role.SYSTEM, "System action")


def test_history_compressed():
    history = SimpleHistory(max_size=2, ttl=600)
    history.init_system_content("System initialized")
    history.insert(Role.USER, "User action 1")
    history.insert(Role.USER, "User action 2")

    entries = history.get()
    assert entries == [
        {"role": "system", "content": "System initialized"},
        {"role": "user", "content": "User action 1"},
        {"role": "user", "content": "User action 2"},
    ]

    history.insert(Role.USER, "User action 3")
    entries = history.get()

    assert entries == [
        {"role": "system", "content": "System initialized"},
        {"role": "user", "content": "User action 2"},
        {"role": "user", "content": "User action 3"},
    ]


def test_history_compressed_no_system():
    history = SimpleHistory(max_size=2, ttl=600)
    history.insert(Role.USER, "User action 1")
    history.insert(Role.USER, "User action 2")

    entries = history.get()
    assert entries == [
        {"role": "user", "content": "User action 1"},
        {"role": "user", "content": "User action 2"},
    ]

    history.insert(Role.USER, "User action 3")
    entries = history.get()

    assert entries == [
        {"role": "user", "content": "User action 2"},
        {"role": "user", "content": "User action 3"},
    ]


def test_str_method():
    history = SimpleHistory(max_size=3, ttl=600)

    # overwrite
    history._SimpleHistory__now = lambda: datetime.datetime(2023, 1, 1)

    history.init_system_content("System initialized")
    history.insert(Role.USER, "User action 1")
    history.insert(Role.ASSISTANT, "Assistant action")
    history.insert(Role.USER, "User action 2")

    expected_str = (
        "2023-01-01 00:00:00, system: System initialized.\n"
        "2023-01-01 00:00:00, user: User action 1.\n"
        "2023-01-01 00:00:00, assistant: Assistant action.\n"
        "2023-01-01 00:00:00, user: User action 2."
    )

    assert str(history) == expected_str
