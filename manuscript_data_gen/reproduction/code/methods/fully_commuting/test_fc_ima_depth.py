import unittest

from traditional_cm_depth import (
    HamiltonianTerm,
    extended_sorted_insertion_fc_groups,
    independent_generators_and_exponents,
    label_to_masks,
    synthesize_group,
)


def term(index: int, label: str, coefficient: float) -> HamiltonianTerm:
    x_mask, z_mask = label_to_masks(label)
    return HamiltonianTerm(index, label, coefficient, x_mask, z_mask)


class TraditionalCMTests(unittest.TestCase):
    def test_extended_sorted_insertion_overlaps_and_seeds_partition(self) -> None:
        terms = [
            term(0, "XX", 4.0),
            term(1, "ZZ", 3.0),
            term(2, "ZI", 2.0),
        ]
        groups, seed_rows = extended_sorted_insertion_fc_groups(terms)
        self.assertEqual(
            [[item.label for item in group] for group in groups],
            [["XX", "ZZ"], ["ZI", "ZZ"]],
        )
        self.assertEqual(seed_rows, [{0, 1}, {2}])
        self.assertEqual(
            sorted(item for rows in seed_rows for item in rows),
            [0, 1, 2],
        )

    def test_generator_exponents_span_dependent_paulis(self) -> None:
        group = [term(0, "XX", 1.0), term(1, "YY", 1.0), term(2, "ZZ", 1.0)]
        generators, exponents = independent_generators_and_exponents(group, 2)
        self.assertEqual(generators, ["XX", "YY"])
        self.assertEqual(exponents, [0b01, 0b10, 0b11])

    def test_yen_clifford_maps_whole_group_to_signed_z_strings(self) -> None:
        group = [
            term(0, "XXI", 1.0),
            term(1, "YYI", 0.5),
            term(2, "ZZI", -0.25),
            term(3, "IIZ", 0.125),
        ]
        metrics, transformed = synthesize_group(group, n_qubits=3, seed=20260828)
        self.assertTrue(metrics["all_terms_z_only"])
        self.assertEqual(len(transformed), len(group))
        self.assertTrue(
            all(set(row["transformed_z_label"]) <= {"I", "Z"} for row in transformed)
        )


if __name__ == "__main__":
    unittest.main()
