"""Capture dashboard screenshots for the README.

Requires (once):
    pip install playwright
    python -m playwright install chromium

Usage:
    # terminal 1
    streamlit run dashboard/app.py
    # terminal 2
    python scripts/screenshots.py

Each page is captured only after Streamlit finishes re-running: we poll for
the spinner to disappear (up to 90s) plus a settle delay, so charts are fully
rendered in the screenshots.

Screenshots land in docs/screenshots/ and are referenced from README.md.
"""

from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8501"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "screenshots"

PAGES = ["Overview", "Funnel", "Loss paths", "Sankey", "Feature compare", "Churn model"]

# Streamlit shows a spinner/status widget while re-running the script; the
# screenshot waits until it is gone so charts are fully rendered.
SPINNER_JS = (
    "() => !document.querySelector('[data-testid=\"stStatusWidget\"], "
    "[data-testid=\"stSpinner\"]')"
)
SPINNER_TIMEOUT_MS = 90_000
SETTLE_MS = 1_500


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def wait_rendered(page) -> None:
    try:
        page.wait_for_function(SPINNER_JS, timeout=SPINNER_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        print("  warn: spinner never disappeared; capturing anyway")
    page.wait_for_timeout(SETTLE_MS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(BASE_URL, wait_until="networkidle")
        wait_rendered(page)
        page.screenshot(path=str(OUT_DIR / f"01_{slug(PAGES[0])}.png"), full_page=True)
        print(f"01 {PAGES[0]}")

        radio_group = page.locator('section[data-testid="stSidebar"] label')
        for i, name in enumerate(PAGES[1:], start=2):
            radio_group.filter(has_text=name).first.click()
            wait_rendered(page)
            page.screenshot(path=str(OUT_DIR / f"{i:02d}_{slug(name)}.png"), full_page=True)
            print(f"{i:02d} {name}")

        browser.close()
    print(f"\nsaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
