def generate_risks(
    predicted_yield,
    soil_ph=None,
    slope=None,
    rainfall=None
):

    risks = []

    # -----------------------------
    # Model yield risk
    # -----------------------------

    if predicted_yield < 1.0:

        risks.append(
            {
                "level": "High",
                "message":
                    "The model predicts a "
                    "relatively low yield "
                    "for this farm scenario."
            }
        )

    elif predicted_yield < 1.5:

        risks.append(
            {
                "level": "Moderate",
                "message":
                    "The model predicts a "
                    "moderate yield."
            }
        )

    # -----------------------------
    # Soil pH
    # -----------------------------

    if soil_ph is not None:

        if soil_ph < 5.5:

            risks.append(
                {
                    "level": "High",
                    "message":
                        "Soil pH is below 5.5."
                }
            )

        elif soil_ph > 7.5:

            risks.append(
                {
                    "level": "Moderate",
                    "message":
                        "Soil pH is above 7.5."
                }
            )

    # -----------------------------
    # Slope
    # -----------------------------

    if slope is not None:

        try:
            slope = float(slope)

            if slope > 8:

                risks.append(
                    {
                        "level": "Moderate",
                        "message":
                            "Steeper terrain may "
                            "increase erosion risk."
                    }
                )

        except Exception:
            pass

    # -----------------------------
    # Rainfall
    # -----------------------------

    if rainfall is not None:

        try:
            rainfall = float(rainfall)

            if rainfall < 400:

                risks.append(
                    {
                        "level": "High",
                        "message":
                            "Available rainfall "
                            "indicators suggest "
                            "elevated water-stress risk."
                    }
                )

        except Exception:
            pass

    return risks
