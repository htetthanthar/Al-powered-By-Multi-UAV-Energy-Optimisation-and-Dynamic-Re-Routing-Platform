"""Offline contract tests; no database, Grafana, or Jenkins writes."""
import collections
import json
import math
from pathlib import Path
import re
import unittest
from unittest.mock import patch, MagicMock
from urllib.parse import parse_qs, urlsplit

from suite_emitter import batch, line_protocol
import suite_emitter
import emit_demo


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.points = batch(1800000000)
        self.groups = collections.defaultdict(list)
        for name, tags, fields, timestamp in self.points:
            self.groups[name].append((tags, fields, timestamp))
        self.boards = [json.loads(p.read_text(encoding="utf-8"))
                       for p in (Path(__file__).parent / "dashboards").glob("*.json")]

    def test_complete_fixture(self):
        self.assertEqual({name: len(rows) for name, rows in self.groups.items()}, {
            "uav_suite_energy": 30, "uav_suite_routing": 60,
            "uav_suite_fit": 150, "uav_suite_accuracy": 1,
            "uav_suite_training": 50, "uav_suite_comparison": 3})
        self.assertEqual(len(line_protocol(self.points).splitlines()), 294)
        self.assertEqual(self.points, batch(1800000000))
        self.assertNotEqual(self.points, batch(1800000010))

    def test_writers_only_target_uav_database(self):
        for module in (suite_emitter, emit_demo):
            for arguments in ([], ["--database", "UAV_Database"]):
                response = MagicMock()
                response.__enter__.return_value.status = 204
                with patch("sys.argv", ["emitter", "--write", *arguments]), \
                        patch.object(module, "urlopen", return_value=response) as send, \
                        patch("builtins.print"):
                    module.main()
                request = send.call_args.args[0]
                self.assertEqual(parse_qs(urlsplit(request.full_url).query)["db"], ["UAV_Database"])
            with patch("sys.argv", ["emitter", "--write", "--database", "other_database"]), \
                    patch.object(module, "urlopen") as send, patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    module.main()
                send.assert_not_called()

    def test_pipeline_database_is_explicit(self):
        for name in ("Jenkinsfile.suite", "Jenkinsfile.demo"):
            content = (Path(__file__).parent / name).read_text(encoding="utf-8")
            self.assertIn("--database UAV_Database", content)

    def test_simulation_labels_and_finite_values(self):
        for name, tags, fields, timestamp in self.points:
            self.assertTrue(name.startswith("uav_suite_"))
            self.assertEqual(tags["source"], "simulation")
            self.assertEqual(timestamp, 1800000000)
            self.assertTrue(all(math.isfinite(value) for value in fields.values()))

    def test_energy_and_savings_are_consistent(self):
        routes = collections.defaultdict(dict)
        for tags, fields, _ in self.groups["uav_suite_energy"]:
            routes[tags["route_id"]][tags["mode"]] = fields
            self.assertAlmostEqual(fields["power_w"], fields["current_a"] * fields["voltage_v"])
            self.assertAlmostEqual(fields["energy_whkm"], fields["power_w"] /
                                   (3.6 * max(2, fields["airspeed_ms"] - fields["headwind_ms"])))
        for modes in routes.values():
            saved = 100 * (1 - modes["optimized"]["energy_whkm"] / modes["baseline"]["energy_whkm"])
            self.assertAlmostEqual(saved, modes["optimized"]["savings_pct"])

    def test_accuracy_is_computed_from_samples(self):
        samples = [fields for _, fields, _ in self.groups["uav_suite_fit"]]
        summary = self.groups["uav_suite_accuracy"][0][1]
        mean = sum(row["physical_w"] for row in samples) / len(samples)
        sse = sum((row["predicted_w"] - row["physical_w"]) ** 2 for row in samples)
        sst = sum((row["physical_w"] - mean) ** 2 for row in samples)
        self.assertAlmostEqual(summary["r_squared"], 1 - sse / sst)
        self.assertAlmostEqual(summary["rmse_w"], math.sqrt(sse / len(samples)))
        self.assertAlmostEqual(summary["mae_w"], sum(abs(row["residual_w"]) for row in samples) / len(samples))

    def test_clearance_flags_and_comparison_bounds(self):
        for _, fields, _ in self.groups["uav_suite_routing"]:
            self.assertEqual(fields["breach"], float(fields["clearance_m"] < 500))
        for _, fields, _ in self.groups["uav_suite_comparison"]:
            self.assertTrue(0 <= fields["violations_pct"] <= 100)
            self.assertEqual(fields["sample_count"], 100)
            self.assertEqual(fields["limit_min"], 55)

    def test_dashboard_contracts(self):
        self.assertEqual(len(self.boards), 7)
        uids = {board["uid"] for board in self.boards}
        self.assertEqual(len(uids), 7)
        panel_count = 0
        for board in self.boards:
            self.assertEqual(board["__inputs"][0]["name"], "DS_UAV_DATABASE")
            self.assertEqual(board["__inputs"][0]["pluginId"], "influxdb")
            self.assertIsNone(board["id"])
            self.assertEqual(board["refresh"], "10s")
            self.assertIn("DEMO", board["title"])
            self.assertEqual(len({p["id"] for p in board["panels"]}), len(board["panels"]))
            self.assertIn("SIMULATED", board["panels"][0]["options"]["content"])
            for link in board["links"]:
                self.assertIn(link["url"].split("?")[0].split("/")[-1], uids)
            for panel in board["panels"]:
                panel_count += 1
                if panel["type"] == "stat":
                    self.assertEqual(panel["options"]["reduceOptions"]["fields"], "")
                for target in panel.get("targets", []):
                    query = target["query"]
                    self.assertEqual(target["datasource"]["uid"], "${DS_UAV_DATABASE}")
                    self.assertEqual(panel["datasource"]["uid"], "${DS_UAV_DATABASE}")
                    self.assertTrue(query.startswith("SELECT "))
                    self.assertIn("time > now() - 30s", query)
                    self.assertIn("\"source\" = 'simulation'", query)
                    measurements = re.findall(r'FROM "(uav_suite_[a-z]+)"', query)
                    self.assertTrue(measurements)
                    for measurement in measurements:
                        self.assertIn(measurement, self.groups)
                        tags, fields, _ = self.groups[measurement][0]
                        allowed = set(tags) | set(fields) | {measurement}
                        for name in re.findall(r'"([a-z_]+)"', query):
                            self.assertIn(name, allowed)
                if panel["type"] == "xychart":
                    for series in panel["options"]["series"]:
                        self.assertEqual(series["frame"]["matcher"], {"id": "byIndex", "options": 0})
                        for axis in ("x", "y"):
                            field = series[axis]["matcher"]["options"]
                            self.assertIn('AS "' + field + '"', panel["targets"][0]["query"])
        self.assertEqual(panel_count, 53)


if __name__ == "__main__":
    unittest.main()
