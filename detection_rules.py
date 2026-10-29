"""
detection_rules.py - rule-based CAN intrusion detection for CAN-Sentinel.

Standalone module (Python standard library only). It does not import the ML
engine or the dashboard. Usage:

    rules = DetectionRules()
    findings = rules.detect(frame)      # list of finding dicts, [] if clean

Every finding has exactly four fields: rule_id, title, severity, reason.

Rules
-----
unknown_can_id          identifier not in the configured baseline map
invalid_payload_length  bad DLC / payload length / byte values
high_message_frequency  too many frames for one identifier in a rolling window
invalid_timestamp       timestamp missing or not a valid "...Z" UTC string
malformed_frame         record is missing fields or has wrong types / ID range

ASSUMPTIONS (not confirmed with the team - see the review notes):
 A1 Baseline IDs 0x100-0x103 are STANDARD (11-bit) IDs. An extended frame with
    the same number is a different identifier and is reported as unknown.
 A2 `dlc` is the number of data bytes (0-8). Classic CAN only; CAN FD and any
    `is_fd` key are not supported and are ignored.
 A3 Error frames carry no meaningful arbitration ID: they skip the unknown-ID,
    ID-range and frequency checks. Their payload and timestamp are still checked.
 A4 Remote frames are counted by the frequency rule, sharing the counter of
    data frames with the same identifier.
 A5 The defaults max_messages=20 / window_seconds=1.0 are PLACEHOLDERS carried
    over from the original code. They are not measured vehicle rates.
 A6 Time comes from the frame `timestamp`, not from arrival time.
 A7 Severity strings are "high" / "medium"; `malformed_frame` uses "medium".
 A8 One baseline map applies to every interface; counters are kept per
    (interface, identifier, standard/extended).
"""

import math
import re
from bisect import bisect_right, insort
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

MAX_STANDARD_ID = 0x7FF        # 11-bit identifier
MAX_EXTENDED_ID = 0x1FFFFFFF   # 29-bit identifier
MAX_DLC = 8                    # classic CAN (assumption A2)

DEFAULT_ALLOWED_IDS = (0x100, 0x101, 0x102, 0x103)

REQUIRED_FIELDS = (
    "interface", "can_id",
    "is_extended_id", "is_remote_frame", "is_error_frame",
)
BOOLEAN_FIELDS = ("is_extended_id", "is_remote_frame", "is_error_frame")

# Accepted timestamp format: YYYY-MM-DDTHH:MM:SS[.fraction]Z (UTC only).
# A strict pattern is used instead of datetime.fromisoformat() because that
# function accepts different formats on different Python versions.
_TIMESTAMP_PATTERN = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})"      # date
    r"T([0-9]{2}):([0-9]{2}):([0-9]{2})"     # time
    r"(?:\.([0-9]{1,9}))?"                   # optional fraction (1-9 digits)
    r"Z"                                      # UTC marker, required
)


def _is_int(value):
    """True for real integers. bool is rejected because True == 1 in Python."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _finding(rule_id, title, severity, reason):
    """Build a finding with exactly the four agreed fields."""
    return {
        "rule_id": rule_id,
        "title": title,
        "severity": severity,
        "reason": reason,
    }


class DetectionRules:
    def __init__(
        self,
        max_messages=20,
        window_seconds=1.0,
        allowed_ids=None,
        allowed_extended_ids=None,
        alert_cooldown_seconds=None,
        max_tracked_identifiers=10000,
    ):
        """
        max_messages            frames allowed per identifier per window
                                (an alert needs MORE than this). Placeholder.
        window_seconds          rolling window length in seconds. Placeholder.
        allowed_ids             standard-ID baseline (default 0x100-0x103).
        allowed_extended_ids    extended-ID baseline (default: none).
        alert_cooldown_seconds  minimum time (frame time) between repeated
                                frequency findings for one identifier.
                                Default = window_seconds. Use 0 to alert on
                                every excess frame (original behaviour).
        max_tracked_identifiers memory safety limit for frequency counters.
        """
        if not _is_int(max_messages) or max_messages < 1:
            raise ValueError("max_messages must be an integer >= 1")
        if not _is_finite_number(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be a finite number > 0")
        try:
            self._window = timedelta(seconds=window_seconds)
        except OverflowError:
            raise ValueError("window_seconds is too large") from None
        if self._window <= timedelta(0):
            raise ValueError("window_seconds must be at least 1 microsecond")

        if alert_cooldown_seconds is None:
            alert_cooldown_seconds = window_seconds
        if (not _is_finite_number(alert_cooldown_seconds)
                or alert_cooldown_seconds < 0):
            raise ValueError("alert_cooldown_seconds must be a number >= 0")
        try:
            self._cooldown = timedelta(seconds=alert_cooldown_seconds)
        except OverflowError:
            raise ValueError("alert_cooldown_seconds is too large") from None

        if not _is_int(max_tracked_identifiers) or max_tracked_identifiers < 1:
            raise ValueError("max_tracked_identifiers must be an integer >= 1")

        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self.max_tracked_identifiers = max_tracked_identifiers
        self.allowed_ids = self._checked_id_set(
            DEFAULT_ALLOWED_IDS if allowed_ids is None else allowed_ids,
            MAX_STANDARD_ID, "allowed_ids")
        self.allowed_extended_ids = self._checked_id_set(
            () if allowed_extended_ids is None else allowed_extended_ids,
            MAX_EXTENDED_ID, "allowed_extended_ids")

        # key -> sorted list of datetimes (frames inside the current window)
        self._times = {}
        # key -> frame time of the last frequency finding (for the cooldown)
        self._last_alert = {}

    @staticmethod
    def _checked_id_set(ids, max_id, name):
        result = set(ids)
        for value in result:
            if not _is_int(value) or not 0 <= value <= max_id:
                raise ValueError(f"{name} must contain integers 0..0x{max_id:X}")
        return result

    def reset(self):
        """Forget all frequency history (use between replays / test runs)."""
        self._times.clear()
        self._last_alert.clear()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def detect(self, frame):
        """Return a list of findings for one frame. Never raises on bad frames."""
        problems = self._structure_problems(frame)
        if problems:
            # Fields cannot be trusted, so no other rule is evaluated.
            return [_finding(
                "malformed_frame",
                "Malformed CAN frame record",
                "medium",
                "The frame record is malformed: " + "; ".join(problems) + ".",
            )]

        findings = []
        can_id = frame["can_id"]
        is_extended = frame["is_extended_id"]
        is_remote = frame["is_remote_frame"]
        is_error = frame["is_error_frame"]
        # Built from the validated number, never copied from `can_id_hex`.
        can_hex = f"0x{can_id:X}"

        # Rule 1: unknown CAN identifier (error frames have no real ID: A3)
        if not is_error and not self._is_known_id(can_id, is_extended):
            kind = "extended" if is_extended else "standard"
            findings.append(_finding(
                "unknown_can_id",
                "Unrecognized CAN identifier",
                "high",
                f"CAN identifier {can_hex} ({kind}) is not in "
                "the configured vehicle signal map.",
            ))

        # Rule 2: invalid payload length or byte values
        payload_problems = self._payload_problems(frame, is_remote)
        if payload_problems:
            findings.append(_finding(
                "invalid_payload_length",
                "Invalid CAN payload",
                "medium",
                f"CAN identifier {can_hex} has an invalid DLC or payload: "
                + "; ".join(payload_problems) + ".",
            ))

        # Timestamp check applies to every frame type.
        timestamp = self._parse_timestamp(frame.get("timestamp"))
        if timestamp is None:
            findings.append(_finding(
                "invalid_timestamp",
                "Invalid CAN timestamp",
                "medium",
                "The frame timestamp is missing or is not a valid "
                "UTC ISO-8601 string ending in Z.",
            ))
        elif not is_error:
            # Rule 3: excessive message frequency (remote frames count: A4)
            key = (frame["interface"], can_id, is_extended)
            finding = self._check_frequency(key, timestamp, can_hex)
            if finding:
                findings.append(finding)

        return findings

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _is_known_id(self, can_id, is_extended):
        if is_extended:
            return can_id in self.allowed_extended_ids
        return can_id in self.allowed_ids

    @staticmethod
    def _structure_problems(frame):
        """List problems with the record itself (not the payload/timestamp)."""
        if not isinstance(frame, Mapping):
            return ["frame is not a dictionary"]

        problems = [f"missing field '{name}'"
                    for name in REQUIRED_FIELDS if name not in frame]
        if problems:
            return problems

        if not isinstance(frame["interface"], str) or not frame["interface"]:
            problems.append("'interface' must be a non-empty string")
        if not _is_int(frame["can_id"]):
            problems.append("'can_id' must be an integer")
        for name in BOOLEAN_FIELDS:
            if not isinstance(frame[name], bool):
                problems.append(f"'{name}' must be true or false")
        if problems:
            return problems

        if frame["is_remote_frame"] and frame["is_error_frame"]:
            problems.append(
                "a frame cannot be both a remote frame and an error frame")
        if not frame["is_error_frame"]:
            if frame["is_extended_id"]:
                limit, kind = MAX_EXTENDED_ID, "extended"
            else:
                limit, kind = MAX_STANDARD_ID, "standard"
            if not 0 <= frame["can_id"] <= limit:
                # The value is not echoed: it is untrusted and may be huge.
                problems.append(
                    f"'can_id' is outside the {kind} range 0x0..0x{limit:X}")
        return problems

    @staticmethod
    def _payload_problems(frame, is_remote):
        """List problems with dlc / data (empty list = payload is valid)."""
        problems = []
        dlc = frame.get("dlc", None)
        data = frame.get("data", None)

        if "dlc" not in frame:
            problems.append("missing 'dlc'")
        elif not _is_int(dlc):
            problems.append("'dlc' is not an integer")
        elif not 0 <= dlc <= MAX_DLC:
            problems.append(f"'dlc' is outside 0..{MAX_DLC}")

        if "data" not in frame:
            problems.append("missing 'data'")
        elif not isinstance(data, (list, tuple, bytes, bytearray)):
            problems.append("'data' is not a list of byte values")
        else:
            if is_remote and len(data) != 0:
                problems.append("remote frame must have empty 'data'")
            if not is_remote and _is_int(dlc) and len(data) != dlc:
                problems.append("length of 'data' does not match 'dlc'")
            if any(not _is_int(b) or not 0 <= b <= 255 for b in data):
                problems.append("'data' has a value that is not a byte (0..255)")
        return problems

    def _parse_timestamp(self, value):
        """Return an aware UTC datetime, or None if `value` is not valid."""
        if not isinstance(value, str):
            return None
        match = _TIMESTAMP_PATTERN.fullmatch(value)
        if match is None:
            return None
        year, month, day, hour, minute, second = (
            int(part) for part in match.groups()[:6])
        fraction = match.group(7) or ""
        microsecond = int(fraction[:6].ljust(6, "0")) if fraction else 0
        try:
            parsed = datetime(year, month, day, hour, minute, second,
                              microsecond, tzinfo=timezone.utc)
            # Headroom check: a timestamp too close to year 1 would overflow
            # when the window is subtracted later.
            parsed - self._window
        except (ValueError, OverflowError):
            return None
        return parsed

    # ------------------------------------------------------------------
    # Rule 3 helper: rolling-window frequency
    # ------------------------------------------------------------------
    def _check_frequency(self, key, timestamp, can_hex):
        times = self._times.get(key)
        if times is None:
            if len(self._times) >= self.max_tracked_identifiers:
                self._forget_idle_identifiers(timestamp)
                if len(self._times) >= self.max_tracked_identifiers:
                    return None          # table full: cannot track a new key
            times = self._times[key] = []

        # The window ends at the NEWEST timestamp seen for this key, so a
        # late (out-of-order) frame cannot drag the window backwards.
        newest = max(times[-1], timestamp) if times else timestamp
        cutoff = newest - self._window
        if timestamp <= cutoff:
            return None                  # too old to belong to this window

        insort(times, timestamp)         # keeps the list sorted
        del times[:bisect_right(times, cutoff)]   # window is (cutoff, newest]

        if len(times) <= self.max_messages:
            return None

        last = self._last_alert.get(key)
        if last is not None and newest - last < self._cooldown:
            return None                  # same flood, already reported
        self._last_alert[key] = newest

        return _finding(
            "high_message_frequency",
            "Excessive CAN message frequency",
            "high",
            f"CAN identifier {can_hex} sent {len(times)} frames within "
            f"{self.window_seconds:g} second(s); the configured limit "
            f"is {self.max_messages}.",
        )

    def _forget_idle_identifiers(self, now):
        """Drop counters whose newest frame is already outside the window."""
        cutoff = now - self._window
        idle = [k for k, times in self._times.items() if times[-1] <= cutoff]
        for key in idle:
            del self._times[key]
            self._last_alert.pop(key, None)