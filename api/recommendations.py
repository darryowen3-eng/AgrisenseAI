def generate_recommendations(
    baseline_yield,
    scenarios
):

    recommendations = []

    if not scenarios:
        return recommendations

    best = scenarios[0]

    best_yield = (
        best["predicted_yield_t_ha"]
    )

    difference = (
        best_yield
        - baseline_yield
    )

    if difference > 0.05:

        recommendations.append(
            f"The model's highest tested "
            f"management scenario is "
            f"{best['scenario']} with a "
            f"predicted yield of "
            f"{best_yield:.2f} t/ha."
        )

        recommendations.append(
            f"This is approximately "
            f"{difference:.2f} t/ha above "
            f"the baseline model prediction."
        )

    else:

        recommendations.append(
            "The tested management scenarios "
            "did not produce a meaningful "
            "increase over the baseline "
            "prediction."
        )

    return recommendations
