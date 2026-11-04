from dataclasses import dataclass
from typing import Optional


@dataclass
class Request:
    req_id: int
    req_type: int
    arrival_time: float
    prompt_tokens: int
    target_gen_tokens: int
    is_priority: bool = False

    gen_tokens: int = 0
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    accum_delay_time: float = 0.0

    def done(self) -> bool:
        return self.gen_tokens >= self.target_gen_tokens
