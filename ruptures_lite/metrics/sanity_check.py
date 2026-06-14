"""Helper to check if two breakpoint lists are comparable.

Ported from ruptures (BSD-2-Clause).
"""

from __future__ import annotations


class BadPartitions(Exception):
    """Exception raised when the partition is bad."""

    pass


def sanity_check(bkps1, bkps2):
    """Check that two partitions are partitions of the same signal.

    Raises:
        BadPartitions: whenever a partition does not respect some conditions.
    """
    for nom, bkps in zip(("first", "second"), (bkps1, bkps2)):
        if len(bkps) == 0:
            raise BadPartitions("The {} partition is empty.".format(nom))
    if max(bkps1) != max(bkps2):
        raise BadPartitions(
            "The end of the last regime is not the same for each of the "
            "partitions:\n{}\n{}".format(bkps1, bkps2)
        )
    for bkps in (bkps1, bkps2):
        seen = set()
        if any(i in seen or seen.add(i) for i in bkps):
            raise BadPartitions("Some indexes are repeated: {}".format(bkps))
