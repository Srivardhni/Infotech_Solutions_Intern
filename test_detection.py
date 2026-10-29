"""
test_detection.py - tests for detection_rules.py (CAN-Sentinel, Member 2).

Run from the folder that contains both files:

    python -m unittest -v test_detection

Only the Python standard library is needed.

Test times are written in milliseconds relative to a fixed start time, so
every result is deterministic (no real clock, no sleeping).
"""

import copy
import random
import unittest
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

from detection_rules import DetectionRules

START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
FREQ = "high_message_frequency"
EXPECTED_FIELDS = {"rule_id", "title", "severity", "reason"}
SEVERITIES = {"low", "medium", "high"}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def ts(ms=0):
    """UTC timestamp string `ms` milliseconds after START, ending in Z."""
    return (START + timedelta(milliseconds=ms)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f") + "Z"


_DEFAULT = object()   # lets a test pass data=None on purpose


def make_frame(can_id=0x100, data=_DEFAULT, ms=0, **overrides):
    """A valid classic-CAN data frame. Use keyword overrides to break it."""
    if data is _DEFAULT:
        data = [0] * 8
    frame = {
        "timestamp": ts(ms),
        "interface": "vcan0",
        "can_id": can_id,
        "can_id_hex": f"0x{can_id:X}" if isinstance(can_id, int) else "n/a",
        "is_extended_id": False,
        "is_remote_frame": False,
        "is_error_frame": False,
        "dlc": len(data) if hasattr(data, "__len__") else 0,
        "data": data,
    }
    frame.update(overrides)
    return frame


def remote_frame(can_id=0x100, dlc=0, ms=0, **overrides):
    return make_frame(can_id, data=[], ms=ms, is_remote_frame=True,
                      dlc=dlc, **overrides)


def error_frame(can_id=0, ms=0, **overrides):
    return make_frame(can_id, ms=ms, is_error_frame=True, **overrides)


def rule_ids(findings):
    return [f["rule_id"] for f in findings]


def feed(rules, frames):
    """Send frames in order; return the list of findings for each frame."""
    return [rules.detect(frame) for frame in frames]


def count_freq(results):
    return sum(FREQ in rule_ids(findings) for findings in results)


class RulesTestCase(unittest.TestCase):
    def setUp(self):
        self.rules = DetectionRules()

    def ids(self, frame):
        return rule_ids(self.rules.detect(frame))


# ----------------------------------------------------------------------
class TestNormalTraffic(RulesTestCase):
    def test_each_baseline_id_is_clean(self):
        for can_id in (0x100, 0x101, 0x102, 0x103):
            with self.subTest(can_id=can_id):
                self.assertEqual(self.rules.detect(make_frame(can_id)), [])

    def test_every_payload_length_0_to_8_is_clean(self):
        for length in range(9):
            with self.subTest(length=length):
                self.assertEqual(
                    self.rules.detect(make_frame(data=[7] * length)), [])

    def test_byte_boundaries_are_valid(self):
        frame = make_frame(data=[0, 255, 1, 254, 0, 255, 127, 128])
        self.assertEqual(self.rules.detect(frame), [])

    def test_steady_traffic_below_limit_is_clean(self):
        # 10 frames per second for 5 seconds, limit is 20 per second.
        results = feed(self.rules, [make_frame(ms=i * 100) for i in range(50)])
        self.assertEqual(results, [[]] * 50)

    def test_instances_do_not_share_state(self):
        other = DetectionRules()
        feed(self.rules, [make_frame(ms=i) for i in range(30)])
        self.assertEqual(other.detect(make_frame(ms=30)), [])


# ----------------------------------------------------------------------
class TestFindingSchema(RulesTestCase):
    def test_findings_have_exactly_the_four_fields(self):
        rules = DetectionRules(max_messages=2)
        findings = []
        findings += rules.detect(make_frame(0x7FF))                 # unknown id
        findings += rules.detect(make_frame(data=[1], dlc=2))       # payload
        findings += rules.detect(make_frame(timestamp="bad"))       # timestamp
        findings += rules.detect(None)                              # malformed
        for i in range(5):                                          # frequency
            findings += rules.detect(make_frame(ms=i))

        self.assertEqual(
            {f["rule_id"] for f in findings},
            {"unknown_can_id", "invalid_payload_length", "invalid_timestamp",
             "malformed_frame", FREQ})
        for finding in findings:
            with self.subTest(rule=finding["rule_id"]):
                self.assertEqual(set(finding), EXPECTED_FIELDS)
                for value in finding.values():
                    self.assertIsInstance(value, str)
                    self.assertTrue(value.strip())
                self.assertIn(finding["severity"], SEVERITIES)

    def test_severities_of_existing_rules_are_unchanged(self):
        rules = DetectionRules(max_messages=1)
        unknown = rules.detect(make_frame(0x7FF, ms=0))
        payload = rules.detect(make_frame(data=[1], dlc=3, ms=1))
        flood = rules.detect(make_frame(ms=2))
        self.assertEqual(unknown[0]["severity"], "high")
        self.assertEqual(payload[0]["severity"], "medium")
        self.assertEqual(flood[0]["severity"], "high")

    def test_returned_findings_are_independent_objects(self):
        first = self.rules.detect(make_frame(0x200))
        first[0]["reason"] = "tampered"
        second = self.rules.detect(make_frame(0x200))
        self.assertNotEqual(second[0]["reason"], "tampered")

    def test_detect_always_returns_a_list(self):
        for frame in (make_frame(), make_frame(0x200), None, {}):
            with self.subTest(frame=frame):
                self.assertIsInstance(self.rules.detect(frame), list)


# ----------------------------------------------------------------------
class TestUnknownId(RulesTestCase):
    def test_unknown_standard_id(self):
        findings = self.rules.detect(make_frame(0x200))
        self.assertEqual(rule_ids(findings), ["unknown_can_id"])
        self.assertEqual(findings[0]["severity"], "high")
        self.assertIn("0x200", findings[0]["reason"])

    def test_highest_valid_standard_id_is_unknown_not_malformed(self):
        self.assertEqual(self.ids(make_frame(0x7FF)), ["unknown_can_id"])

    def test_id_zero_is_valid_but_unknown(self):
        self.assertEqual(self.ids(make_frame(0)), ["unknown_can_id"])

    def test_extended_id_with_baseline_number_is_unknown(self):
        # Assumption A1: baseline IDs are standard IDs.
        findings = self.rules.detect(make_frame(0x100, is_extended_id=True))
        self.assertEqual(rule_ids(findings), ["unknown_can_id"])
        self.assertIn("extended", findings[0]["reason"])

    def test_highest_valid_extended_id_is_unknown_not_malformed(self):
        frame = make_frame(0x1FFFFFFF, is_extended_id=True)
        self.assertEqual(self.ids(frame), ["unknown_can_id"])

    def test_extended_baseline_can_be_configured(self):
        rules = DetectionRules(allowed_extended_ids={0x18FF0001})
        frame = make_frame(0x18FF0001, is_extended_id=True)
        self.assertEqual(rules.detect(frame), [])

    def test_custom_standard_baseline(self):
        rules = DetectionRules(allowed_ids={0x200})
        self.assertEqual(rules.detect(make_frame(0x200)), [])
        self.assertEqual(rule_ids(rules.detect(make_frame(0x100))),
                         ["unknown_can_id"])

    def test_empty_baseline_makes_every_id_unknown(self):
        rules = DetectionRules(allowed_ids=set())
        self.assertEqual(rule_ids(rules.detect(make_frame(0x100))),
                         ["unknown_can_id"])

    def test_reason_uses_the_numeric_id_not_the_frame_hex_text(self):
        frame = make_frame(0x7FF, can_id_hex="0x100")
        reason = self.rules.detect(frame)[0]["reason"]
        self.assertIn("0x7FF", reason)
        self.assertNotIn("0x100", reason)

    def test_untrusted_frame_text_is_not_copied_into_reasons(self):
        frame = make_frame(0x200, can_id_hex="<script>alert(1)</script>",
                           interface="<b>vcan0</b>")
        reason = self.rules.detect(frame)[0]["reason"]
        self.assertNotIn("<script>", reason)
        self.assertNotIn("<b>", reason)

    def test_missing_can_id_hex_is_fine(self):
        frame = make_frame(0x100)
        del frame["can_id_hex"]
        self.assertEqual(self.rules.detect(frame), [])

    def test_unknown_id_is_reported_on_every_frame(self):
        results = feed(self.rules, [make_frame(0x200, ms=i) for i in range(3)])
        self.assertEqual([rule_ids(r)[0] for r in results],
                         ["unknown_can_id"] * 3)


# ----------------------------------------------------------------------
class TestPayload(RulesTestCase):
    def test_dlc_larger_than_data(self):
        self.assertEqual(self.ids(make_frame(data=[1, 2, 3, 4], dlc=8)),
                         ["invalid_payload_length"])

    def test_dlc_smaller_than_data(self):
        self.assertEqual(self.ids(make_frame(data=[0] * 8, dlc=4)),
                         ["invalid_payload_length"])

    def test_dlc_above_8_is_invalid_for_classic_can(self):
        for length in (9, 12, 64):
            with self.subTest(length=length):
                frame = make_frame(data=[0] * length)   # dlc == len(data)
                self.assertEqual(self.ids(frame), ["invalid_payload_length"])

    def test_negative_dlc(self):
        self.assertEqual(self.ids(make_frame(data=[], dlc=-1)),
                         ["invalid_payload_length"])

    def test_dlc_of_wrong_type(self):
        for bad in ("8", None, 8.0, True, [8], {}):
            with self.subTest(dlc=bad):
                self.assertEqual(self.ids(make_frame(dlc=bad)),
                                 ["invalid_payload_length"])

    def test_missing_dlc_or_data(self):
        for name in ("dlc", "data"):
            with self.subTest(missing=name):
                frame = make_frame()
                del frame[name]
                self.assertEqual(self.ids(frame), ["invalid_payload_length"])

    def test_data_of_wrong_type(self):
        for bad in (None, "ABCDEFGH", {"a": 1}, 12345, object()):
            with self.subTest(data=type(bad).__name__):
                self.assertEqual(self.ids(make_frame(data=bad, dlc=8)),
                                 ["invalid_payload_length"])

    def test_byte_values_out_of_range_or_wrong_type(self):
        for bad in (256, -1, 1000, "1", 1.5, True, None, [1]):
            with self.subTest(byte=bad):
                frame = make_frame(data=[0] * 7 + [bad])
                self.assertEqual(self.ids(frame), ["invalid_payload_length"])

    def test_bytes_bytearray_and_tuple_payloads_are_accepted(self):
        for data in (bytes(8), bytearray(8), tuple([1] * 8)):
            with self.subTest(type=type(data).__name__):
                self.assertEqual(self.rules.detect(make_frame(data=data)), [])

    def test_fd_style_frames_are_not_supported(self):
        # The old code let an `is_fd` key relax the limit to 64 bytes.
        for length in (12, 64):
            with self.subTest(length=length):
                frame = make_frame(data=[0] * length, is_fd=True)
                self.assertEqual(self.ids(frame), ["invalid_payload_length"])

    def test_payload_and_unknown_id_are_both_reported(self):
        frame = make_frame(0x200, data=[1], dlc=4)
        self.assertEqual(self.ids(frame),
                         ["unknown_can_id", "invalid_payload_length"])

    def test_reason_names_the_problem(self):
        reason = self.rules.detect(make_frame(data=[1], dlc=4))[0]["reason"]
        self.assertIn("does not match", reason)
        reason = self.rules.detect(make_frame(data=[0] * 7 + [300]))[0]["reason"]
        self.assertIn("not a byte", reason)


# ----------------------------------------------------------------------
class TestFrameTypes(RulesTestCase):
    # ---- remote frames ------------------------------------------------
    def test_valid_remote_frames_for_every_dlc_0_to_8(self):
        for dlc in range(9):
            with self.subTest(dlc=dlc):
                self.assertEqual(self.rules.detect(remote_frame(dlc=dlc)), [])

    def test_remote_frame_with_data_is_invalid(self):
        frame = remote_frame(dlc=2)
        frame["data"] = [1, 2]
        self.assertEqual(self.ids(frame), ["invalid_payload_length"])

    def test_remote_frame_with_bad_dlc_does_not_crash(self):
        for bad in (9, -1, "8", None, 2.0, True):
            with self.subTest(dlc=bad):
                self.assertEqual(self.ids(remote_frame(dlc=bad)),
                                 ["invalid_payload_length"])

    def test_remote_frame_with_unknown_id_is_reported(self):
        self.assertEqual(self.ids(remote_frame(0x200)), ["unknown_can_id"])

    def test_remote_frame_with_invalid_timestamp_is_reported(self):
        frame = remote_frame(timestamp="garbage")
        self.assertEqual(self.ids(frame), ["invalid_timestamp"])

    # ---- error frames -------------------------------------------------
    def test_error_frame_is_not_reported_as_unknown_id(self):
        # Assumption A3: error frames have no meaningful arbitration ID.
        self.assertEqual(self.rules.detect(error_frame(can_id=0)), [])

    def test_error_frame_id_outside_normal_range_is_tolerated(self):
        self.assertEqual(self.rules.detect(error_frame(can_id=0x20000004)), [])

    def test_error_frame_with_inconsistent_payload_is_reported(self):
        frame = error_frame(data=[1, 2], dlc=8)
        self.assertEqual(self.ids(frame), ["invalid_payload_length"])

    def test_error_frame_with_invalid_timestamp_is_reported(self):
        self.assertEqual(self.ids(error_frame(timestamp=None)),
                         ["invalid_timestamp"])

    def test_frame_cannot_be_remote_and_error(self):
        frame = make_frame(data=[], is_remote_frame=True, is_error_frame=True)
        self.assertEqual(self.ids(frame), ["malformed_frame"])

    # ---- standard vs extended ----------------------------------------
    def test_valid_extended_frame_has_no_structural_finding(self):
        frame = make_frame(0x12345678, is_extended_id=True)
        self.assertEqual(self.ids(frame), ["unknown_can_id"])


# ----------------------------------------------------------------------
class TestMalformedFrames(RulesTestCase):
    def assertMalformed(self, frame):
        self.assertEqual(self.ids(frame), ["malformed_frame"])

    def test_frame_that_is_not_a_dictionary(self):
        for bad in (None, [], "frame", 42, 3.14, (1, 2), b"x"):
            with self.subTest(frame=bad):
                self.assertMalformed(bad)

    def test_read_only_mapping_is_accepted(self):
        self.assertEqual(self.rules.detect(MappingProxyType(make_frame())), [])

    def test_empty_dictionary(self):
        self.assertMalformed({})

    def test_each_required_field_missing(self):
        for name in ("interface", "can_id", "is_extended_id",
                     "is_remote_frame", "is_error_frame"):
            with self.subTest(missing=name):
                frame = make_frame()
                del frame[name]
                self.assertMalformed(frame)
                self.assertIn(name, self.rules.detect(frame)[0]["reason"])

    def test_can_id_of_wrong_type(self):
        for bad in (None, "256", "0x100", 256.0, True, False, [256], {}, (256,)):
            with self.subTest(can_id=bad):
                self.assertMalformed(make_frame(can_id=bad))

    def test_flags_of_wrong_type(self):
        for name in ("is_extended_id", "is_remote_frame", "is_error_frame"):
            for bad in ("false", "true", 0, 1, None, []):
                with self.subTest(field=name, value=bad):
                    self.assertMalformed(make_frame(**{name: bad}))

    def test_interface_of_wrong_type_or_empty(self):
        for bad in (None, "", 5, ["vcan0"]):
            with self.subTest(interface=bad):
                self.assertMalformed(make_frame(interface=bad))

    def test_can_id_out_of_range(self):
        self.assertMalformed(make_frame(-1))
        self.assertMalformed(make_frame(0x800))                       # > 11 bit
        self.assertMalformed(make_frame(0x20000000, is_extended_id=True))
        self.assertMalformed(make_frame(-1, is_extended_id=True))

    def test_huge_can_id_does_not_crash(self):
        frame = make_frame(1 << 20000)
        self.assertMalformed(frame)

    def test_standard_range_check_uses_the_extended_flag(self):
        # 0x800 is invalid as a standard ID but valid as an extended ID.
        self.assertEqual(self.ids(make_frame(0x800, is_extended_id=True)),
                         ["unknown_can_id"])

    def test_malformed_frame_stops_other_rules(self):
        frame = make_frame(data=[1], dlc=9, timestamp="bad")
        del frame["can_id"]
        self.assertMalformed(frame)

    def test_extra_unknown_fields_are_ignored(self):
        frame = make_frame(is_fd=False, extra="x", sequence=5)
        self.assertEqual(self.rules.detect(frame), [])

    def test_malformed_reason_does_not_echo_untrusted_values(self):
        reason = self.rules.detect(make_frame(can_id="<script>"))[0]["reason"]
        self.assertNotIn("<script>", reason)


# ----------------------------------------------------------------------
class TestTimestamp(RulesTestCase):
    def test_valid_timestamp_formats(self):
        for value in ("2026-01-01T12:00:00Z",
                      "2026-01-01T12:00:00.123Z",
                      "2026-01-01T12:00:00.123456Z",
                      "2026-01-01T12:00:00.123456789Z",
                      "9999-12-31T23:59:59Z"):
            with self.subTest(timestamp=value):
                self.assertEqual(
                    DetectionRules().detect(make_frame(timestamp=value)), [])

    def test_invalid_timestamp_strings(self):
        for value in ("", "Z", "not a timestamp", "2026-13-45T99:99:99Z",
                      "2026-02-30T00:00:00Z", "2026-01-01T24:00:00Z",
                      "2026-01-01T00:00:00",              # no Z (naive)
                      "2026-01-01T05:30:00+05:30",        # offset, not Z
                      "2026-01-01T00:00:00+00:00",        # +00:00, not Z
                      "2026-01-01T00:00:00+05:30Z",       # offset AND Z
                      "2026-01-01T00:00:00z",             # lowercase z
                      "2026-01-01T00:00:00ZZ",
                      "2026-01-01T00:00:00 Z",
                      "2026-01-01T00:00:00Z\n",
                      " 2026-01-01T00:00:00Z",
                      "2026-01-01Z",                      # date only
                      "2026-01-01T00:00Z",                # no seconds
                      "20260101T000000Z",                 # basic format
                      "2026-1-1T0:0:0Z",                  # not zero padded
                      "2026-01-01T00:00:00.Z",            # empty fraction
                      "0000-01-01T00:00:00Z",             # year 0
                      "0001-01-01T00:00:00Z"):            # overflows window math
            with self.subTest(timestamp=value):
                self.assertEqual(
                    self.ids(make_frame(timestamp=value)), ["invalid_timestamp"])

    def test_non_string_timestamps(self):
        for value in (None, 1700000000, 1.7e9, True, [], {}, b"2026-01-01T00:00:00Z"):
            with self.subTest(timestamp=value):
                self.assertEqual(
                    self.ids(make_frame(timestamp=value)), ["invalid_timestamp"])

    def test_missing_timestamp(self):
        frame = make_frame()
        del frame["timestamp"]
        self.assertEqual(self.ids(frame), ["invalid_timestamp"])

    def test_invalid_timestamp_frames_are_not_counted(self):
        results = feed(self.rules,
                       [make_frame(timestamp="bad") for _ in range(40)])
        self.assertEqual(count_freq(results), 0)
        self.assertEqual(self.rules.detect(make_frame(ms=0)), [])

    def test_reason_does_not_echo_the_bad_timestamp(self):
        reason = self.rules.detect(make_frame(timestamp="<script>"))[0]["reason"]
        self.assertNotIn("<script>", reason)


# ----------------------------------------------------------------------
class TestFrequency(RulesTestCase):
    def test_exactly_the_limit_is_clean_and_one_more_alerts(self):
        results = feed(self.rules, [make_frame(ms=i) for i in range(21)])
        self.assertEqual(count_freq(results[:20]), 0)
        self.assertEqual(rule_ids(results[20]), [FREQ])
        self.assertEqual(results[20][0]["severity"], "high")
        self.assertIn("21", results[20][0]["reason"])

    def test_steady_traffic_exactly_at_the_limit_is_clean(self):
        # Regression: 20 frames per second (every 50 ms) at limit 20.
        # The old code raised 81 false alerts here.
        results = feed(self.rules, [make_frame(ms=i * 50) for i in range(101)])
        self.assertEqual(count_freq(results), 0)

    def test_21_frames_inside_one_window_alert(self):
        results = feed(self.rules, [make_frame(ms=i * 49) for i in range(21)])
        self.assertEqual(count_freq(results), 1)

    def test_window_slides_forward(self):
        frames = ([make_frame(ms=i) for i in range(20)]
                  + [make_frame(ms=2000 + i) for i in range(20)])
        self.assertEqual(count_freq(feed(self.rules, frames)), 0)

    def test_custom_limit_and_window(self):
        rules = DetectionRules(max_messages=3, window_seconds=0.5)
        fast = feed(rules, [make_frame(ms=i * 100) for i in range(4)])
        self.assertEqual(count_freq(fast), 1)
        rules = DetectionRules(max_messages=3, window_seconds=0.5)
        slow = feed(rules, [make_frame(ms=i * 200) for i in range(4)])
        self.assertEqual(count_freq(slow), 0)

    def test_burst_gives_one_finding_not_one_per_frame(self):
        # Regression: the old code emitted 180 findings for this burst.
        results = feed(self.rules, [make_frame(ms=i) for i in range(200)])
        self.assertEqual(count_freq(results), 1)

    def test_sustained_flood_reports_once_per_window(self):
        results = feed(self.rules, [make_frame(ms=i * 10) for i in range(350)])
        alert_times = [i * 10 for i, r in enumerate(results) if FREQ in rule_ids(r)]
        self.assertEqual(alert_times, [200, 1200, 2200, 3200])

    def test_cooldown_zero_alerts_on_every_excess_frame(self):
        rules = DetectionRules(alert_cooldown_seconds=0)
        results = feed(rules, [make_frame(ms=i) for i in range(30)])
        self.assertEqual(count_freq(results), 10)        # frames 21..30

    def test_new_flood_after_a_quiet_period_alerts_again(self):
        frames = ([make_frame(ms=i) for i in range(30)]
                  + [make_frame(ms=10000 + i) for i in range(30)])
        self.assertEqual(count_freq(feed(self.rules, frames)), 2)

    def test_counters_are_per_identifier(self):
        frames = []
        for i in range(15):
            frames += [make_frame(0x100, ms=i), make_frame(0x101, ms=i)]
        self.assertEqual(count_freq(feed(self.rules, frames)), 0)

    def test_counters_are_per_interface(self):
        frames = []
        for i in range(15):
            frames += [make_frame(ms=i, interface="vcan0"),
                       make_frame(ms=i, interface="vcan1")]
        self.assertEqual(count_freq(feed(self.rules, frames)), 0)
        more = [make_frame(ms=15 + i, interface="vcan0") for i in range(6)]
        self.assertEqual(count_freq(feed(self.rules, more)), 1)

    def test_standard_and_extended_ids_are_counted_separately(self):
        frames = []
        for i in range(15):
            frames += [make_frame(0x100, ms=i),
                       make_frame(0x100, ms=i, is_extended_id=True)]
        self.assertEqual(count_freq(feed(self.rules, frames)), 0)

    def test_remote_frames_are_counted(self):
        # Regression: the old code ignored remote frames, hiding RTR floods.
        results = feed(self.rules, [remote_frame(ms=i) for i in range(21)])
        self.assertEqual(count_freq(results), 1)

    def test_remote_and_data_frames_share_one_counter(self):
        frames = [make_frame(ms=i) if i % 2 else remote_frame(ms=i)
                  for i in range(21)]
        self.assertEqual(count_freq(feed(self.rules, frames)), 1)

    def test_error_frames_are_not_counted(self):
        results = feed(self.rules, [error_frame(ms=i) for i in range(100)])
        self.assertEqual(count_freq(results), 0)
        after = feed(self.rules, [make_frame(ms=100 + i) for i in range(20)])
        self.assertEqual(count_freq(after), 0)

    def test_unknown_id_flood_is_also_counted(self):
        results = feed(self.rules, [make_frame(0x7FF, ms=i) for i in range(21)])
        unknown = sum("unknown_can_id" in rule_ids(r) for r in results)
        self.assertEqual(unknown, 21)
        self.assertEqual(count_freq(results), 1)

    def test_frames_with_bad_payload_still_count(self):
        frames = [make_frame(data=[1], dlc=4, ms=i) for i in range(21)]
        self.assertEqual(count_freq(feed(self.rules, frames)), 1)

    def test_identical_timestamps_are_all_counted(self):
        results = feed(self.rules, [make_frame(ms=0) for _ in range(21)])
        self.assertEqual(count_freq(results), 1)

    def test_reason_reports_count_window_and_limit(self):
        rules = DetectionRules(max_messages=2, window_seconds=0.5)
        reason = feed(rules, [make_frame(ms=i) for i in range(3)])[2][0]["reason"]
        self.assertIn("0x100", reason)
        self.assertIn("3 frames", reason)
        self.assertIn("0.5", reason)
        self.assertIn("limit is 2", reason)


# ----------------------------------------------------------------------
class TestTimestampOrdering(RulesTestCase):
    def test_late_frame_inside_the_window_is_counted(self):
        frames = [make_frame(ms=1000 + i) for i in range(20)]
        frames.append(make_frame(ms=1005))              # arrives late
        results = feed(self.rules, frames)
        self.assertEqual(count_freq(results[:20]), 0)
        self.assertEqual(rule_ids(results[20]), [FREQ])

    def test_stale_frame_is_ignored_and_does_not_inflate_the_count(self):
        # Regression: the old code appended the stale frame, so the 20th
        # real frame produced a false alert.
        results = feed(self.rules, [make_frame(ms=10000 + i) for i in range(19)])
        self.assertEqual(count_freq(results), 0)
        self.assertEqual(self.rules.detect(make_frame(ms=1000)), [])   # stale
        self.assertEqual(self.rules.detect(make_frame(ms=10019)), [])  # 20th

    def test_stale_frame_gets_no_finding_even_during_a_flood(self):
        # Alert suppression is switched off so the only thing that can keep
        # the stale frame quiet is the "too old for this window" rule.
        rules = DetectionRules(alert_cooldown_seconds=0)
        results = feed(rules, [make_frame(ms=10000 + i) for i in range(25)])
        self.assertEqual(count_freq(results), 5)             # frames 21..25
        self.assertEqual(rules.detect(make_frame(ms=1000)), [])

    def test_window_pruning_still_correct_after_a_late_frame(self):
        rules = DetectionRules(max_messages=2)
        results = feed(rules, [make_frame(ms=1000), make_frame(ms=1010),
                               make_frame(ms=1005)])            # late, count 3
        self.assertEqual(rule_ids(results[2]), [FREQ])
        later = feed(rules, [make_frame(ms=2011), make_frame(ms=2012)])
        self.assertEqual(count_freq(later), 0)                  # old ones gone

    def test_frames_arriving_in_descending_order_still_alert(self):
        frames = [make_frame(ms=i) for i in range(20, -1, -1)]
        results = feed(self.rules, frames)
        self.assertEqual(count_freq(results[:20]), 0)
        self.assertEqual(rule_ids(results[20]), [FREQ])

    def test_forward_time_jump_clears_the_window(self):
        frames = ([make_frame(ms=i) for i in range(20)]
                  + [make_frame(ms=3600000 + i) for i in range(20)])
        self.assertEqual(count_freq(feed(self.rules, frames)), 0)

    def test_backward_then_forward_jump_does_not_crash(self):
        frames = [make_frame(ms=3600000), make_frame(ms=0),
                  make_frame(ms=3600010), make_frame(ms=5)]
        self.assertEqual(feed(self.rules, frames), [[], [], [], []])

    def test_identical_frame_replayed_inside_the_window_counts_duplicates(self):
        frame = make_frame(ms=500)
        results = feed(self.rules, [frame] * 21)
        self.assertEqual(count_freq(results), 1)

    def test_replayed_capture_with_old_timestamps_is_ignored(self):
        # DOCUMENTED LIMITATION: frames older than the window (relative to
        # the newest frame seen) are not counted. Replaying a capture
        # through the same instance is therefore invisible until reset().
        capture = ([make_frame(ms=i) for i in range(25)]
                   + [make_frame(ms=5000)])
        self.assertEqual(count_freq(feed(self.rules, capture)), 1)
        self.assertEqual(count_freq(feed(self.rules, capture[:25])), 0)

    def test_reset_makes_a_replayed_capture_detectable_again(self):
        capture = [make_frame(ms=i) for i in range(25)]
        self.assertEqual(count_freq(feed(self.rules, capture)), 1)
        self.rules.reset()
        self.assertEqual(count_freq(feed(self.rules, capture)), 1)


# ----------------------------------------------------------------------
class TestConfiguration(RulesTestCase):
    def test_invalid_max_messages(self):
        for bad in (0, -1, "20", 2.5, None, True, [20]):
            with self.subTest(max_messages=bad):
                with self.assertRaises(ValueError):
                    DetectionRules(max_messages=bad)

    def test_invalid_window_seconds(self):
        for bad in (0, -1, -0.5, "1", None, True, float("nan"),
                    float("inf"), 1e30, 1e-9):
            with self.subTest(window_seconds=bad):
                with self.assertRaises(ValueError):
                    DetectionRules(window_seconds=bad)

    def test_invalid_cooldown(self):
        for bad in (-1, "1", True, float("nan"), float("inf")):
            with self.subTest(cooldown=bad):
                with self.assertRaises(ValueError):
                    DetectionRules(alert_cooldown_seconds=bad)

    def test_invalid_tracking_limit(self):
        for bad in (0, -1, "10", 1.5, True):
            with self.subTest(limit=bad):
                with self.assertRaises(ValueError):
                    DetectionRules(max_tracked_identifiers=bad)

    def test_invalid_id_sets(self):
        for bad in ([-1], [0x800], ["256"], [True], [1.5], [None]):
            with self.subTest(allowed_ids=bad):
                with self.assertRaises(ValueError):
                    DetectionRules(allowed_ids=bad)
        with self.assertRaises(ValueError):
            DetectionRules(allowed_extended_ids=[0x20000000])
        with self.assertRaises(TypeError):
            DetectionRules(allowed_ids=5)

    def test_valid_unusual_settings(self):
        DetectionRules(max_messages=1, window_seconds=2)
        DetectionRules(window_seconds=0.001, alert_cooldown_seconds=0)

    def test_defaults_match_the_original_placeholders(self):
        self.assertEqual(self.rules.max_messages, 20)
        self.assertEqual(self.rules.window_seconds, 1.0)
        self.assertEqual(self.rules.allowed_ids, {0x100, 0x101, 0x102, 0x103})

    def test_reset_clears_frequency_history(self):
        feed(self.rules, [make_frame(ms=i) for i in range(20)])
        self.rules.reset()
        results = feed(self.rules, [make_frame(ms=i) for i in range(20)])
        self.assertEqual(count_freq(results), 0)

    def test_tracking_limit_ignores_new_identifiers_when_full(self):
        # Documented fail-open: when the table is full of recently active
        # identifiers, a new one is not counted (unknown IDs are still
        # reported by the unknown-ID rule).
        rules = DetectionRules(allowed_ids={0x100, 0x101, 0x102},
                               max_messages=1, max_tracked_identifiers=2)
        feed(rules, [make_frame(0x100, ms=0), make_frame(0x101, ms=0)])
        results = feed(rules, [make_frame(0x102, ms=500),
                               make_frame(0x102, ms=501)])
        self.assertEqual(count_freq(results), 0)

    def test_tracking_limit_frees_idle_identifiers(self):
        rules = DetectionRules(allowed_ids={0x100, 0x101, 0x102},
                               max_messages=1, max_tracked_identifiers=2)
        feed(rules, [make_frame(0x100, ms=0), make_frame(0x101, ms=0)])
        results = feed(rules, [make_frame(0x102, ms=5000),
                               make_frame(0x102, ms=5001)])
        self.assertEqual(count_freq(results), 1)

    def test_many_distinct_identifiers_stay_bounded(self):
        rules = DetectionRules(max_tracked_identifiers=100)
        for i in range(1000):
            rules.detect(make_frame(0x10000 + i, is_extended_id=True, ms=i * 2000))
        self.assertLessEqual(len(rules._times), 100)


# ----------------------------------------------------------------------
class TestRobustness(RulesTestCase):
    def test_detect_does_not_modify_the_input_frame(self):
        frame = make_frame(0x200, data=[1, 2], dlc=5)
        before = copy.deepcopy(frame)
        self.rules.detect(frame)
        self.assertEqual(frame, before)

    def test_random_garbage_never_crashes_and_keeps_the_schema(self):
        rng = random.Random(20260101)
        garbage = [
            None, "", "x", "2026-01-01T12:00:00Z", 0, -1, 1, 8, 255, 256,
            0x7FF, 0x800, 2 ** 31, 2 ** 64, 1.5, float("nan"), float("inf"),
            True, False, [], [1], [0] * 8, [0] * 9, [300] * 8, {}, {"a": 1},
            (), (1, 2), b"\x01", bytearray(8), object(),
        ]
        rules = DetectionRules(max_messages=3, max_tracked_identifiers=5)
        for step in range(3000):
            frame = make_frame(rng.choice([0x100, 0x101, 0x200, 0x7FF]),
                               ms=rng.randint(0, 3000))
            for key in list(frame):
                roll = rng.random()
                if roll < 0.15:
                    frame[key] = rng.choice(garbage)
                elif roll < 0.20:
                    del frame[key]
            if rng.random() < 0.05:
                frame = rng.choice(garbage)
            findings = rules.detect(frame)       # must not raise
            self.assertIsInstance(findings, list)
            for finding in findings:
                self.assertEqual(set(finding), EXPECTED_FIELDS)
                self.assertTrue(all(isinstance(v, str) for v in finding.values()))
                self.assertIn(finding["severity"], SEVERITIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)