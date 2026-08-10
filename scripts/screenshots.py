"""Capture dashboard screenshots for the README.

Requires (once):
    pip install playwright
    python -m playwright install chromium

Usage:
    # terminal 1
    streamlit run dashboard/app.py
    # terminal 2
    python scripts/screenshots.py

Robustness: after clicking a sidebar section we (1) wait for Streamlit's
re-run spinner to appear, (2) wait for it to disappear, (3) wait for a
marker text unique to the target page, and (4) retry the click once if the
marker never shows — so a screenshot can never capture the previous page.

Screenshots land in docs/screenshots/ and are referenced from README.md.
"""

from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8501"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "screenshots"

PAGES = ["Overview", "Funnel", "Loss paths", "Sankey", "Feature compare", "Churn model"]

# Unique text marker per section, verified before capturing.
PAGE_MARKERS = {
    "Overview": "Behavior type distribution",
    "Funnel": "Granularity",
    "Loss paths": "Loss-path patterns among non-buying sessions",
    "Sankey": "Show start/end nodes",
    "Feature compare": "Metric",
    "Churn model": "Decision threshold",
}

SPINNER_SEL = '[data-testid="stStatusWidget"], [data-testid="stSpinner"]'
SPINNER_PRESENT_JS = f"() => document.querySelector('{SPINNER_SEL}') !== null"
SPINNER_GONE_JS = f"() => document.querySelector('{SPINNER_SEL}') === null"


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def wait_rendered(page, name: str, retries: int = 2) -> None:
    marker = PAGE_MARKERS[name]
    for attempt in range(retries + 1):
        try:
            # 1) click took effect: a re-run spinner appears
            try:
                page.wait_for_function(SPINNER_PRESENT_JS, timeout=8_000)
            except PlaywrightTimeoutError:
                pass  # re-run may be too fast to observe; keep going
            # 2) re-run finished: spinner gone
            page.wait_for_function(SPINNER_GONE_JS, timeout=90_000)
            # 3) target page really rendered
            page.get_by_text(marker, exact=False).first.wait_for(timeout=30_000)
            page.wait_for_timeout(1_200)
            return
        except PlaywrightTimeoutError:
            if attempt < retries:
                print(f"  retry {attempt + 1}/{retries}: marker '{marker}' missing, re-clicking")
                page.locator('section[data-testid="stSidebar"] label').filter(
                    has_text=name
                ).first.click()
            else:
                raise


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(BASE_URL, wait_until="networkidle")
        wait_rendered(page, PAGES[0])
        page.screenshot(path=str(OUT_DIR / f"01_{slug(PAGES[0])}.png"), full_page=True)
        print(f"01 {PAGES[0]} (marker verified)")

        radio_group = page.locator('section[data-testid="stSidebar"] label')
        for i, name in enumerate(PAGES[1:], start=2):
            radio_group.filter(has_text=name).first.click()
            wait_rendered(page, name)
            page.screenshot(path=str(OUT_DIR / f"{i:02d}_{slug(name)}.png"), full_page=True)
            print(f"{i:02d} {name} (marker verified)")

        browser.close()
    print(f"\nsaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
