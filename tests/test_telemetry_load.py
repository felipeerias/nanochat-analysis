import unittest

import pandas as pd

from loader.telemetry_load import (
    arm,
    certified,
    direction_verdict_by_step,
    metric_certifier,
)


def row(metric, step, value=1.0, *, defined=True, arm_name="shadow_fp32"):
    return {
        "metric": metric,
        "step": step,
        "value_scalar": value,
        "is_defined": defined,
        "acceptance_arm": arm_name,
    }


class CertificationTests(unittest.TestCase):
    def setUp(self):
        rows = []
        verdicts = {
            "random": (0.0, 2.0),
            "gradient": (0.0, 1.0),
            "update": (2.0, 0.0),
        }
        for direction, values in verdicts.items():
            for step, value in enumerate(values, 1):
                rows.append(row(f"curvature/verdict_code_{direction}", step, value))
        for step in (1, 2):
            rows.extend([
                row("curvature/shadow_verdict_code", step, 2.0),
                row("update/p1", step),
                row("update/actual", step),
                row("curvature/gHg", step, defined=step != 1),
                row("curvature/e_sym_gradient", step),
                row("curvature/dhd", step),
                row("update/p2", step),
                row("curvature/e_sym_update", step),
                row("curvature/e_sym_random", step),
            ])
        self.sparse = pd.DataFrame(rows)

    def selected_steps(self, frame, name):
        return frame.loc[frame["metric"].eq(name), "step"].tolist()

    def test_each_metric_uses_its_actual_dependency(self):
        selected = certified(self.sparse, "shadow_fp32")
        self.assertEqual(self.selected_steps(selected, "update/p1"), [1, 2])
        self.assertEqual(self.selected_steps(selected, "update/actual"), [1, 2])
        self.assertEqual(self.selected_steps(selected, "curvature/gHg"), [1])
        self.assertEqual(self.selected_steps(selected, "curvature/e_sym_gradient"), [1])
        self.assertEqual(self.selected_steps(selected, "curvature/dhd"), [2])
        self.assertEqual(self.selected_steps(selected, "update/p2"), [2])
        self.assertEqual(self.selected_steps(selected, "curvature/e_sym_update"), [2])
        self.assertEqual(self.selected_steps(selected, "curvature/e_sym_random"), [1])

    def test_verdict_and_no_hvp_rows_are_not_hidden_by_failed_verdicts(self):
        selected = certified(self.sparse, "shadow_fp32")
        self.assertEqual(
            self.selected_steps(selected, "curvature/verdict_code_gradient"),
            [1, 2],
        )
        self.assertEqual(
            self.selected_steps(selected, "curvature/shadow_verdict_code"),
            [1, 2],
        )

    def test_definedness_remains_an_explicit_filter(self):
        selected = certified(self.sparse, "shadow_fp32")
        ghg = selected.loc[selected["metric"].eq("curvature/gHg")]
        self.assertEqual(len(ghg), 1)
        self.assertFalse(bool(ghg.iloc[0]["is_defined"]))

    def test_direction_verdicts_are_arm_scoped(self):
        native = pd.concat([
            self.sparse,
            pd.DataFrame([
                row("curvature/verdict_code_gradient", 9, 0.0, arm_name="native")
            ]),
        ], ignore_index=True)
        self.assertEqual(
            direction_verdict_by_step(native, "gradient", "native"),
            {9: "passed"},
        )
        self.assertTrue(arm(native, "native")["acceptance_arm"].eq("native").all())

    def test_unknown_schema_metric_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no certification rule"):
            metric_certifier("curvature/future_quantity")

    def test_unknown_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown HVP direction"):
            direction_verdict_by_step(self.sparse, "sideways", "shadow_fp32")


if __name__ == "__main__":
    unittest.main()
