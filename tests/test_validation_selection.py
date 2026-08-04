import unittest

from utils.selection import compare_metric_dicts, normalize_selection_policy


class ValidationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.policy = normalize_selection_policy(
            {
                "primary": {"metric": "dice", "mode": "max", "tolerance": 0.001},
                "tie_breakers": [
                    {"metric": "missing_prediction_rate", "mode": "min"},
                    {"metric": "hd95", "mode": "min"},
                    {"metric": "assd", "mode": "min"},
                    {"metric": "boundary_f1", "mode": "max"},
                ],
            }
        )

    def test_surface_candidate_can_win_inside_predeclared_dice_tolerance(self):
        incumbent = {
            "dice": 0.8760,
            "missing_prediction_rate": 0.0,
            "hd95": 7.30,
            "assd": 2.58,
            "boundary_f1": 0.864,
        }
        candidate = {
            "dice": 0.8752,
            "missing_prediction_rate": 0.0,
            "hd95": 7.15,
            "assd": 2.47,
            "boundary_f1": 0.862,
        }
        selected, _ = compare_metric_dicts(candidate, incumbent, self.policy)
        self.assertTrue(selected)

    def test_candidate_outside_dice_tolerance_cannot_win_on_surface(self):
        incumbent = {
            "dice": 0.8760,
            "missing_prediction_rate": 0.0,
            "hd95": 7.30,
            "assd": 2.58,
            "boundary_f1": 0.864,
        }
        candidate = {
            "dice": 0.8740,
            "missing_prediction_rate": 0.0,
            "hd95": 6.00,
            "assd": 2.00,
            "boundary_f1": 0.870,
        }
        selected, _ = compare_metric_dicts(candidate, incumbent, self.policy)
        self.assertFalse(selected)

    def test_missing_prediction_rate_precedes_surface_tie_breakers(self):
        incumbent = {
            "dice": 0.8755,
            "missing_prediction_rate": 0.0,
            "hd95": 7.5,
            "assd": 2.6,
            "boundary_f1": 0.86,
        }
        candidate = {
            "dice": 0.8758,
            "missing_prediction_rate": 0.01,
            "hd95": 6.0,
            "assd": 2.0,
            "boundary_f1": 0.87,
        }
        selected, _ = compare_metric_dicts(candidate, incumbent, self.policy)
        self.assertFalse(selected)


if __name__ == "__main__":
    unittest.main()
