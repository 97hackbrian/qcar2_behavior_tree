from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class Status(Enum):
    RUNNING = 'RUNNING'
    SUCCESS = 'SUCCESS'
    FAILURE = 'FAILURE'


@dataclass
class TickContext:
    now_sec: float
    blackboard: dict[str, Any]
    publish_state: Callable[[str], None]


class BTNode:
    def __init__(self, name: str):
        self.name = name

    def tick(self, context: TickContext) -> Status:
        raise NotImplementedError

    def reset(self) -> None:
        return


class Sequence(BTNode):
    def __init__(self, name: str, children: list[BTNode]):
        super().__init__(name)
        self.children = children
        self.current_index = 0

    def tick(self, context: TickContext) -> Status:
        while self.current_index < len(self.children):
            child = self.children[self.current_index]
            status = child.tick(context)
            if status == Status.RUNNING:
                return Status.RUNNING
            if status == Status.FAILURE:
                return Status.FAILURE
            self.current_index += 1
        return Status.SUCCESS

    def reset(self) -> None:
        self.current_index = 0
        for child in self.children:
            child.reset()


class Selector(BTNode):
    def __init__(self, name: str, children: list[BTNode]):
        super().__init__(name)
        self.children = children
        self.current_index = 0

    def tick(self, context: TickContext) -> Status:
        while self.current_index < len(self.children):
            child = self.children[self.current_index]
            status = child.tick(context)
            if status == Status.RUNNING:
                return Status.RUNNING
            if status == Status.SUCCESS:
                return Status.SUCCESS
            self.current_index += 1
        return Status.FAILURE

    def reset(self) -> None:
        self.current_index = 0
        for child in self.children:
            child.reset()


class RepeatForever(BTNode):
    def __init__(self, name: str, child: BTNode):
        super().__init__(name)
        self.child = child

    def tick(self, context: TickContext) -> Status:
        status = self.child.tick(context)
        if status in (Status.SUCCESS, Status.FAILURE):
            self.child.reset()
        return Status.RUNNING

    def reset(self) -> None:
        self.child.reset()


class Condition(BTNode):
    def __init__(self, name: str, predicate: Callable[[TickContext], bool]):
        super().__init__(name)
        self.predicate = predicate

    def tick(self, context: TickContext) -> Status:
        return Status.SUCCESS if self.predicate(context) else Status.FAILURE


class Action(BTNode):
    def __init__(self, name: str, callback: Callable[[TickContext], Status]):
        super().__init__(name)
        self.callback = callback

    def tick(self, context: TickContext) -> Status:
        return self.callback(context)


class Wait(BTNode):
    def __init__(self, name: str, seconds: float):
        super().__init__(name)
        self.seconds = max(0.0, float(seconds))
        self.started_at: float | None = None

    def tick(self, context: TickContext) -> Status:
        if self.started_at is None:
            self.started_at = context.now_sec
            context.publish_state(f'{self.name}: waiting {self.seconds:.2f}s')
        elapsed = context.now_sec - self.started_at
        if elapsed >= self.seconds:
            self.started_at = None
            return Status.SUCCESS
        return Status.RUNNING

    def reset(self) -> None:
        self.started_at = None
