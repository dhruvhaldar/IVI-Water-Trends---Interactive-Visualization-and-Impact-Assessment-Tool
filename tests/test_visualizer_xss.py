import pytest
import pandas as pd
import re
from ivi_water.visualizer import WaterTrendsVisualizer


class TestVisualizerXSS:
    """Test suite for XSS prevention in visualizations."""

    def setup_method(self):
        self.viz = WaterTrendsVisualizer()
        self.malicious_payload = "<img src=x onerror=alert(1)>"
        self.malicious_location = f"V001-{self.malicious_payload}"
        self.safe_location = "V001-&lt;img src=x onerror=alert(1)&gt;"

        # Sample data with malicious content
        # Ensure we have both intervention statuses (0 and 1) for comparison plots
        self.df = pd.DataFrame(
            {
                "location_id": [self.malicious_location] * 4,
                "year": [2020, 2021, 2022, 2020],
                "season": ["monsoon", "monsoon", "monsoon", "winter"],
                "water_area_ha": [100.0, 110.0, 105.0, 90.0],
                "water_body_count": [5, 6, 5, 4],
                "pond_presence": [1, 1, 0, 0],  # Mix of 1 and 0
                "crop_yield_ton_per_ha": [2.5, 2.6, 2.7, 2.0],
            }
        )

    def test_sanitize_text_helper(self):
        """Verify the helper method correctly escapes HTML."""
        sanitized = self.viz._sanitize_text(self.malicious_payload)
        assert sanitized == "&lt;img src=x onerror=alert(1)&gt;"
        assert self.malicious_payload not in sanitized

        # Test None handling
        assert self.viz._sanitize_text(None) is None

        # Test regular text
        assert self.viz._sanitize_text("Safe Text") == "Safe Text"

    def test_seasonal_chart_title_xss(self):
        """Verify seasonal chart title is sanitized."""
        # Case 1: Auto-generated title from malicious location_id
        fig = self.viz.create_seasonal_stacked_area_chart(
            self.df, location_id=self.malicious_location
        )
        layout = fig.to_dict()["layout"]
        title = layout["title"]["text"]

        assert self.malicious_payload not in title
        assert "&lt;img" in title

        # Case 2: Custom malicious title
        fig = self.viz.create_seasonal_stacked_area_chart(
            self.df, location_id=self.malicious_location, title=self.malicious_payload
        )
        layout = fig.to_dict()["layout"]
        title = layout["title"]["text"]

        assert title == "&lt;img src=x onerror=alert(1)&gt;"

    def test_multi_location_dashboard_xss(self):
        """Verify dashboard titles and trace names are sanitized."""
        fig = self.viz.create_multi_location_dashboard(
            self.df, location_ids=[self.malicious_location]
        )

        # Check main title
        layout = fig.to_dict()["layout"]
        title = layout["title"]["text"]
        assert self.malicious_payload not in title
        assert "&lt;img" in title

        # Check trace names
        for trace in fig.data:
            if "name" in trace:
                assert self.malicious_payload not in trace["name"]
                if self.malicious_location in self.df["location_id"].values:
                    pass

    def test_other_charts_xss(self):
        """Verify other chart types sanitize titles."""
        # Comparison plot
        fig = self.viz.create_comparison_line_plot(
            self.df, title=self.malicious_payload
        )
        assert fig.layout.title.text == "&lt;img src=x onerror=alert(1)&gt;"

        # Distribution plot
        fig = self.viz.create_water_body_distribution(
            self.df, title=self.malicious_payload
        )
        assert fig.layout.title.text == "&lt;img src=x onerror=alert(1)&gt;"

        # Heatmap
        fig = self.viz.create_trend_heatmap(self.df, title=self.malicious_payload)
        assert fig.layout.title.text == "&lt;img src=x onerror=alert(1)&gt;"

        # Scatter
        fig = self.viz.create_intervention_impact_scatter(
            self.df, title=self.malicious_payload
        )
        assert fig.layout.title.text == "&lt;img src=x onerror=alert(1)&gt;"

    def test_html_output_does_not_contain_payload(self):
        """End-to-end test: verify generated HTML is safe."""
        fig = self.viz.create_seasonal_stacked_area_chart(
            self.df, location_id=self.malicious_location
        )
        html_output = fig.to_html()

        # The payload should NOT appear unescaped
        assert self.malicious_payload not in html_output
        assert "&lt;img src=x onerror=alert(1)&gt;" in html_output
