from __future__ import annotations

from dataclasses import dataclass


REQUEST_TYPE_SERVICE: int = 0
REQUEST_TYPE_CONTENT: int = 1
REQUEST_TYPE_ENERGY: int = 2


@dataclass(slots=True)
class Request:
    """Explicit request structure used across the environment.

    Deadline and priority are meaningful for service requests only.
    Content and emergency energy requests keep default placeholder values.
    """

    req_type: int
    req_size: int
    req_id: int
    deadline: float = 0.0
    priority: int = 0

    @property
    def is_service(self) -> bool:
        return self.req_type == REQUEST_TYPE_SERVICE

    @property
    def is_content(self) -> bool:
        return self.req_type == REQUEST_TYPE_CONTENT

    @property
    def is_energy(self) -> bool:
        return self.req_type == REQUEST_TYPE_ENERGY

    @classmethod
    def service(cls, req_size: int, req_id: int, deadline: float, priority: int) -> "Request":
        return cls(
            req_type=REQUEST_TYPE_SERVICE,
            req_size=req_size,
            req_id=req_id,
            deadline=deadline,
            priority=priority,
        )

    @classmethod
    def content(cls, req_id: int) -> "Request":
        return cls(req_type=REQUEST_TYPE_CONTENT, req_size=0, req_id=req_id)

    @classmethod
    def energy(cls) -> "Request":
        return cls(req_type=REQUEST_TYPE_ENERGY, req_size=0, req_id=0)
