import abc
import dataclasses as dt
import datetime
import enum


class Role(enum.StrEnum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


@dt.dataclass
class HistoryEntry:
    role: Role
    content: str
    timestamp: datetime.datetime

    def get_representation(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class HistoryInterface(abc.ABC):
    @abc.abstractmethod
    def init_system_content(self, content: str) -> None:
        pass

    @abc.abstractmethod
    def insert(self, role: Role, content: str) -> None:
        pass

    @abc.abstractmethod
    def get(self) -> list[dict[str, str]]:
        pass


class SimpleHistory(HistoryInterface):
    def __init__(self, max_size: int, ttl: int) -> None:
        self._history: list[HistoryEntry] = []
        self._system_content: str | None = None
        self._max_size = max_size
        self._ttl = ttl

    def init_system_content(self, content: str) -> None:
        self._system_content = content

    @staticmethod
    def __now() -> datetime.datetime:
        return datetime.datetime.now()

    def __get_system_entry(self) -> HistoryEntry | None:
        if self._system_content is None:
            return None
        return HistoryEntry(Role.SYSTEM, self._system_content, self.__now())

    def __compress(self) -> None:
        to_remove = len(self._history) - self._max_size
        if to_remove > 0:
            del self._history[:to_remove]

    def __remove_expired(self) -> None:
        new_history = []
        for item in self._history:
            if item.timestamp + datetime.timedelta(seconds=self._ttl) > self.__now():
                new_history.append(item)
        self._history = new_history

    def insert(self, role: Role, content: str) -> None:
        if role == Role.SYSTEM:
            raise ValueError("Can't insert system role entry. Use init_system_content function.")
        entry = HistoryEntry(role=role, content=content, timestamp=self.__now())
        self._history.append(entry)
        self.__compress()

    def __get_combined_history(self) -> list[HistoryEntry]:
        system_entry = self.__get_system_entry()
        if system_entry is None:
            return self._history
        else:
            return [system_entry] + self._history

    def get(self) -> list[dict[str, str]]:
        self.__remove_expired()
        history = self.__get_combined_history()
        return [item.get_representation() for item in history]

    def __str__(self) -> str:
        history = self.__get_combined_history()
        entries = [f"{item.timestamp}, {item.role}: {item.content}." for item in history]
        return "\n".join(entries)
