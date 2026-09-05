"""
Single source of truth for turning model/graph/rule signals into a
final risk score and an advisory action. Both live scoring
(src/scoring.py) and the dashboard's historical view call
combine_scores() so there is exactly one formula for the final risk
score anywhere in this project.
"""

MODEL_WEIGHT = 0.60
GRAPH_WEIGHT = 0.25
RULE_WEIGHT = 0.15

REVIEW_THRESHOLD = 0.50
STEP_UP_THRESHOLD = 0.80


def combine_scores(model_score: float, graph_score: float, rule_score: float) -> float:
    blended = (
        MODEL_WEIGHT * model_score
        + GRAPH_WEIGHT * graph_score
        + RULE_WEIGHT * rule_score
    )
    return min(max(blended, 0.0), 1.0)


def choose_action(final_score: float) -> str:
    if final_score >= STEP_UP_THRESHOLD:
        return "step_up"

    if final_score >= REVIEW_THRESHOLD:
        return "review"

    return "allow"
