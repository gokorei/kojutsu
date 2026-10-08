"""Network-address predicates shared across entry points.

A leaf module: stdlib only. The loopback check lived in three places --
the webhook URL validator, the dev console bind check and the lifecycle --
which disagreed about bracket-stripping order and case handling. A loopback
test that depends on which caller asks is not a test.
"""

import ipaddress

__all__ = ["is_loopback_host"]


def is_loopback_host(host: str) -> bool:
    """True when a hostname binds or points at this machine."""
    candidate = host.strip().removeprefix("[").removesuffix("]").lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False
