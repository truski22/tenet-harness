"""Everything that constrains what an agent may do during a run, bundled for the tool executor."""
from dataclasses import dataclass

from ..policy import Policy
from ..redaction import Redactor
from ..storage import RunStore
from .approvals import ApprovalBroker
from .loops import LoopPolicy


@dataclass(frozen=True)
class Governance:
    policy: Policy
    approvals: ApprovalBroker
    store: RunStore
    redact: Redactor
    loops: LoopPolicy | None
