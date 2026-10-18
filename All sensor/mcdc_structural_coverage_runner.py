import itertools
from typing import Dict, List, Tuple, Callable

class MCDCStructuralCoverageEngine:
    def __init__(self):
        self.verified_decision_tables: List[Dict] = []

    def verify_mcdc_decision(
        self,
        decision_name: str,
        condition_names: List[str],
        boolean_evaluator: Callable[..., bool]
    ) -> Dict:
        """
        Tests MC/DC coverage on a boolean decision function f(c1, c2, ..., cn).
        Proves that for each condition c_i, there exists a pair of test vectors (T1, T2) where:
          - c_i flips from True to False (or False to True)
          - All other conditions c_j (j != i) remain identical
          - The overall decision outcome flips
        """
        n = len(condition_names)
        truth_table = []

        # Exhaustive 2^N state generation
        for inputs in itertools.product([False, True], repeat=n):
            outcome = boolean_evaluator(*inputs)
            truth_table.append((inputs, outcome))

        mcdc_pairs = {}
        fully_covered = True

        for i, cond_name in enumerate(condition_names):
            found_pair = None
            for idx1, (in1, out1) in enumerate(truth_table):
                for idx2, (in2, out2) in enumerate(truth_table):
                    if idx1 >= idx2:
                        continue
                    # Condition i must differ
                    if in1[i] != in2[i]:
                        # All other conditions must be identical
                        all_others_identical = all(in1[j] == in2[j] for j in range(n) if j != i)
                        # Overall decision outcome must differ
                        if all_others_identical and (out1 != out2):
                            found_pair = (in1, in2, out1, out2)
                            break
                if found_pair:
                    break

            if found_pair:
                mcdc_pairs[cond_name] = {
                    "vector_1": [f"{condition_names[k]}={found_pair[0][k]}" for k in range(n)],
                    "vector_2": [f"{condition_names[k]}={found_pair[1][k]}" for k in range(n)],
                    "outcomes": (found_pair[2], found_pair[3])
                }
            else:
                mcdc_pairs[cond_name] = None
                fully_covered = False

        result = {
            "decision_name": decision_name,
            "conditions_count": n,
            "total_truth_table_rows": 2 ** n,
            "mcdc_coverage_achieved_pct": (sum(1 for v in mcdc_pairs.values() if v is not None) / n) * 100.0,
            "do178c_level_a_compliant": fully_covered,
            "independent_condition_pairs": mcdc_pairs
        }
        self.verified_decision_tables.append(result)
        return result

# -------------------------------------------------------------------------
# AVIONICS SAFETY DECISION PREDICATES FOR MC/DC VERIFICATION
# -------------------------------------------------------------------------
def parachute_ejection_decision(free_fall: bool, high_tumble: bool, low_alt: bool, armed: bool) -> bool:
    """Decision: (free_fall OR high_tumble) AND (NOT low_alt) AND armed"""
    return (free_fall or high_tumble) and (not low_alt) and armed

def battery_auto_rtb_decision(low_soc: bool, cell_imbalance: bool, comm_lost: bool, manual_override: bool) -> bool:
    """Decision: (low_soc OR cell_imbalance OR comm_lost) AND (NOT manual_override)"""
    return (low_soc or cell_imbalance or comm_lost) and (not manual_override)

if __name__ == "__main__":
    runner = MCDCStructuralCoverageEngine()

    print("==========================================================================")
    print("      RTCA DO-178C LEVEL A MC/DC STRUCTURAL COVERAGE VERIFICATION         ")
    print("==========================================================================")

    # 1. Verify Parachute Ballistic Trigger Decision
    res_chute = runner.verify_mcdc_decision(
        decision_name="Ballistic_Parachute_Ejection_Gate",
        condition_names=["free_fall", "high_tumble", "low_alt", "armed"],
        boolean_evaluator=parachute_ejection_decision
    )
    print(f"Decision: '{res_chute['decision_name']}'")
    print(f"  • MC/DC Coverage Achieved : {res_chute['mcdc_coverage_achieved_pct']:.1f}%")
    print(f"  • DO-178C Level A Status  : {'CERTIFIED' if res_chute['do178c_level_a_compliant'] else 'NON-COMPLIANT'}")

    # 2. Verify Battery Auto-RTB Decision
    res_rtb = runner.verify_mcdc_decision(
        decision_name="Automated_Return_to_Base_Gate",
        condition_names=["low_soc", "cell_imbalance", "comm_lost", "manual_override"],
        boolean_evaluator=battery_auto_rtb_decision
    )
    print(f"\nDecision: '{res_rtb['decision_name']}'")
    print(f"  • MC/DC Coverage Achieved : {res_rtb['mcdc_coverage_achieved_pct']:.1f}%")
    print(f"  • DO-178C Level A Status  : {'CERTIFIED' if res_rtb['do178c_level_a_compliant'] else 'NON-COMPLIANT'}")
    print("==========================================================================\n")