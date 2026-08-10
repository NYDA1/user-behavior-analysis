"""Capture dashboard screenshots for the README.

Requires (once):
    pip install playwright
    python -m playwright install chromium

Usage:
    # terminal 1
    streamlit run dashboard/app.py
    # terminal 2
    python scripts/screenshots.py

Screenshots land in docs/screenshots/ and are referenced from README.md.
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8501"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "screenshots"

PAGES = ["Overview", "Funnel", "Loss paths", "Sankey", "Feature compare", "Churn model"]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(BASE_URL, wait_until="networkidle")
        page.wait_for_timeout(2000)
        page.screenshot(path=str(OUT_DIR / f"01_{slug(PAGES[0])}.png"), full_page=True)
        print(f"01 {PAGES[0]}")

        radio_group = page.locator('section[data-testid="stSidebar"] label')
        for i, name in enumerate(PAGES[1:], start=2):
            radio_group.filter(has_text=name).first.click()
            page.wait_for_timeout(2000)
            page.screenshot(path=str(OUT_DIR / f"{i:02d}_{slug(name)}.png"), full_page=True)
            print(f"{i:02d} {name}")

        browser.close()
    print(f"\nsaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
