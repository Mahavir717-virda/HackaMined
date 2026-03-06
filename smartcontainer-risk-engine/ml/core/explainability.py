"""
Explainability Engine
=====================
Generates human-readable explanations for risk predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class RiskExplainer:
    """Generate explainable natural language explanations."""

    def __init__(self):
        self.rules = {
            'weight_diff': {
                'high': ("Large weight discrepancy detected between declared and measured.", 
                        "This may indicate container tampering or documentation error."),
                'low': ("Minor weight variance within expected tolerances.", 
                       "Standard measurement deviation is acceptable.")
            },
            'value_density': {
                'high': ("Unusually high value-to-weight ratio detected.", 
                        "This may indicate high-value smuggling risk or luxury contraband."),
                'low': ("Suspiciously low value for weight declared.", 
                       "This may suggest commodity fraud or undervaluation."),
                'normal': ("Value-to-weight ratio within normal ranges.")
            },
            'route': {
                'high_risk': ("Container routing through high-risk jurisdictions.", 
                             "Enhanced scrutiny recommended."),
                'unusual': ("Unusual or infrequent trade route detected.", 
                           "Historical data limited for this route."),
                'normal': ("Standard trade route with normal frequency.")
            },
            'dwell': {
                'excessive': ("Container excessive dwell time in port.", 
                             "Extended detention may indicate customs hold or processing delays."),
                'minimal': ("Unusually short dwell time in port.", 
                           "Rapid clearance may warrant verification."),
                'normal': ("Dwell time within expected range.")
            },
            'time': {
                'off_hours': ("Shipment declared outside business hours.", 
                             "May indicate intentional avoidance of regular oversight."),
                'weekend': ("Weekend submission detected.", 
                           "May indicate attempt to bypass weekday procedures."),
                'normal': ("Standard business hours submission.")
            },
            'hs_code': {
                'restricted': ("HS code associated with controlled/restricted items.", 
                              "Enhanced verification required."),
                'normal': ("Standard commodity classification.")
            },
            'import_history': {
                'flagged': ("Importer has history of flagged shipments.", 
                           "Heightened risk profile warranted."),
                'clean': ("Importer has clean compliance history.")
            }
        }

    def _assess_weight_issue(self, row: pd.Series) -> str:
        """Assess weight-related anomalies."""
        weight_diff_pct = row.get('weight_diff_pct', 0)
        abs_diff_pct = abs(weight_diff_pct)

        if abs_diff_pct > 20:
            direction = "higher" if weight_diff_pct > 0 else "lower"
            return f"Measured weight is {abs_diff_pct:.1f}% {direction} than declared."
        return None

    def _assess_value_density(self, row: pd.Series) -> str:
        """Assess value-to-weight anomalies."""
        if row.get('flag_high_value_density', 0):
            vpk = row.get('value_per_kg', 0)
            return f"Exceptionally high value density (${vpk:,.0f}/kg) - possible luxury or contraband goods."
        elif row.get('flag_low_value_density', 0):
            vpk = row.get('value_per_kg', 0)
            return f"Suspiciously low value density (${vpk:.2f}/kg) - possible undervaluation or commodity fraud."
        return None

    def _assess_route_risk(self, row: pd.Series) -> str:
        """Assess route-based risk factors."""
        factors = []

        origin_risk = row.get('origin_country_risk', 0)
        if origin_risk == 2:
            factors.append("high-risk origin country")
        elif origin_risk == 1:
            factors.append("medium-risk origin country")

        dest_risk = row.get('dest_country_risk', 0)
        if dest_risk == 2:
            factors.append("high-risk destination")

        route_freq = row.get('route_frequency', 1)
        if route_freq < 3:
            factors.append("infrequent or unusual trade route")

        if factors:
            return "Route factors: " + ", ".join(factors) + "."
        return None

    def _assess_dwell_time(self, row: pd.Series) -> str:
        """Assess dwell time anomalies."""
        if row.get('flag_excessive_dwell', 0):
            dwell = row.get('Dwell_Time_Hours', 0)
            return f"Excessive port dwell time ({dwell:.0f} hours) - may indicate customs hold or processing issues."
        elif row.get('flag_minimal_dwell', 0):
            dwell = row.get('Dwell_Time_Hours', 0)
            return f"Unusually rapid clearance ({dwell:.0f} hours) - warrants verification."
        return None

    def _assess_timing(self, row: pd.Series) -> str:
        """Assess submission timing anomalies."""
        factors = []

        if row.get('is_night', 0):
            factors.append("submitted during off-hours")
        if row.get('is_weekend', 0):
            factors.append("weekend submission")

        if factors:
            return "Timing flags: " + ", ".join(factors) + " (may indicate avoidance of oversight)."
        return None

    def _assess_hs_code(self, row: pd.Series) -> str:
        """Assess HS code risk classification."""
        hs_risk = row.get('hs_code_risk', 0)
        
        high_risk_codes = {'27', '28', '39', '84', '85', '86'}
        hs_prefix = str(row.get('HS_Code', '00'))[:2]
        
        if hs_prefix in high_risk_codes:
            return f"HS code {hs_prefix}** associated with controlled/restricted commodities."
        return None

    def generate_explanation(self, row: pd.Series, risk_level: str, 
                           risk_score: float) -> str:
        """
        Generate comprehensive explanation for prediction.

        Args:
            row: Feature row from DataFrame
            risk_level: Classification (Critical/High/Medium/Low)
            risk_score: Risk probability (0-1)

        Returns:
            Explanation string
        """
        explanations = []
        
        # Assess different categories
        weight_issue = self._assess_weight_issue(row)
        if weight_issue:
            explanations.append(weight_issue)

        value_issue = self._assess_value_density(row)
        if value_issue:
            explanations.append(value_issue)

        route_issue = self._assess_route_risk(row)
        if route_issue:
            explanations.append(route_issue)

        dwell_issue = self._assess_dwell_time(row)
        if dwell_issue:
            explanations.append(dwell_issue)

        timing_issue = self._assess_timing(row)
        if timing_issue:
            explanations.append(timing_issue)

        hs_issue = self._assess_hs_code(row)
        if hs_issue:
            explanations.append(hs_issue)

        # Format final explanation
        if risk_level == 'Critical' and explanations:
            return " ".join(explanations[:2])  # Top 2 factors
        elif risk_level in ('High', 'Medium') and explanations:
            return explanations[0]
        elif explanations:
            return "Minor risk indicators noted: " + "; ".join(explanations[:2]) + "."
        else:
            return f"All parameters within normal ranges. Risk score: {risk_score:.2f}. Standard processing."

    def generate_batch_explanations(self, df: pd.DataFrame, risk_levels: np.ndarray,
                                   risk_scores: np.ndarray) -> List[str]:
        """
        Generate explanations for batch of predictions.

        Args:
            df: Feature DataFrame
            risk_levels: Array of risk levels
            risk_scores: Array of risk scores

        Returns:
            List of explanation strings
        """
        explanations = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            level = risk_levels[idx]
            score = risk_scores[idx]
            expl = self.generate_explanation(row, level, score)
            explanations.append(expl)
        
        logger.info(f"Generated explanations for {len(explanations)} predictions")
        return explanations
