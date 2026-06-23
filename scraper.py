"""
scraper.py
----------
Scrapes data science job postings from WeWorkRemotely.
Run this first to collect raw data before cleaning.

Requirements:
    pip install requests beautifulsoup4 pandas
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import json
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "https://weworkremotely.com"

# Pages to scrape (data-related categories on WWR)
CATEGORY_URLS = [
    "/categories/remote-data-science-jobs",
    "/categories/remote-programming-jobs",    # contains ML engineer roles
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Polite delay between requests (seconds) — don't remove this, it's respectful
DELAY_MIN = 2
DELAY_MAX = 4


# ── Helper Functions ───────────────────────────────────────────────────────────

def polite_sleep():
    """Wait a random amount between requests — avoids hammering the server."""
    delay = random.uniform(DELAY_MIN, DELAY_MAX)
    print(f"  Waiting {delay:.1f}s...")
    time.sleep(delay)


def fetch_page(url):
    """
    Fetch a URL and return a BeautifulSoup object.
    Returns None if the request fails.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()   # raises an error for 4xx/5xx responses
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"  ERROR fetching {url}: {e}")
        return None


def scrape_listing_page(category_url):
    """
    Scrape all job listings from a category page.
    Returns a list of dicts with basic info (title, company, url, location, date).
    """
    full_url = BASE_URL + category_url
    print(f"\nScraping category: {full_url}")
    soup = fetch_page(full_url)

    if not soup:
        return []

    jobs = []

    # WWR wraps each job in a <li> inside <section class="jobs">
    job_listings = soup.select("section.jobs ul li.feature, section.jobs ul li")

    print(f"  Found {len(job_listings)} listings on page")

    for li in job_listings:
        # Skip spacer rows and ads
        if "view-all" in li.get("class", []):
            continue

        anchor = li.find("a", href=True)
        if not anchor:
            continue

        job_url = BASE_URL + anchor["href"]

        # Title and company are in specific divs inside the anchor
        title_tag = li.find("span", class_="title")
        company_tag = li.find("span", class_="company")
        region_tag = li.find("span", class_="region")
        date_tag = li.find("time")

        jobs.append({
            "title": title_tag.get_text(strip=True) if title_tag else None,
            "company": company_tag.get_text(strip=True) if company_tag else None,
            "location_raw": region_tag.get_text(strip=True) if region_tag else None,
            "posted_date_raw": date_tag.get("datetime") if date_tag else None,
            "job_url": job_url,
            "scraped_at": datetime.now().isoformat(),
        })

    return jobs


def scrape_job_detail(job_url):
    """
    Visit an individual job posting page and extract:
    - Full description text (for skill extraction later)
    - Salary info (often buried in description or a dedicated field)
    - Tags/skills listed by the poster
    """
    print(f"  Scraping detail: {job_url}")
    soup = fetch_page(job_url)

    if not soup:
        return {"description": None, "salary_raw": None, "tags_raw": None}

    # Description is in a div with class "listing-container"
    desc_div = soup.find("div", class_="listing-container")
    description = desc_div.get_text(separator=" ", strip=True) if desc_div else None

    # Tags (skills) are listed as <li> inside <ul class="tags">
    tags_ul = soup.find("ul", class_="tags")
    tags = None
    if tags_ul:
        tag_items = tags_ul.find_all("li")
        tags = ", ".join(t.get_text(strip=True) for t in tag_items)

    # Salary: WWR doesn't always have a dedicated field; we capture it from
    # the description and let the cleaning step extract it
    salary_raw = None
    if description:
        import re
        # Look for salary patterns like "$120,000", "$80k-$120k", "£50,000"
        salary_match = re.search(
            r"[\$£€]\s?[\d,]+[k]?(?:\s?[-–]\s?[\$£€]?\s?[\d,]+[k]?)?",
            description,
            re.IGNORECASE,
        )
        if salary_match:
            salary_raw = salary_match.group(0)

    return {
        "description": description,
        "salary_raw": salary_raw,
        "tags_raw": tags,
    }


# ── Main Scraping Loop ─────────────────────────────────────────────────────────

def run_scraper(max_jobs=200):
    """
    Main function. Scrapes listing pages, then visits each job detail page.
    Saves raw data to raw_jobs.json and raw_jobs.csv.

    Args:
        max_jobs: Stop after collecting this many jobs (to avoid overloading)
    """
    all_jobs = []

    # Step 1: collect all listing-level info across categories
    for category_url in CATEGORY_URLS:
        listings = scrape_listing_page(category_url)
        all_jobs.extend(listings)
        polite_sleep()

        if len(all_jobs) >= max_jobs:
            break

    all_jobs = all_jobs[:max_jobs]
    print(f"\nTotal listings collected: {len(all_jobs)}")

    # Step 2: visit each job page to get description + salary + tags
    for i, job in enumerate(all_jobs):
        print(f"\n[{i+1}/{len(all_jobs)}]")
        detail = scrape_job_detail(job["job_url"])
        job.update(detail)   # merge detail fields into the job dict
        polite_sleep()

    # Step 3: save raw data — NEVER discard the raw data, save it first
    print("\nSaving raw data...")

    # JSON (preserves all types, good for backup)
    with open("raw_jobs.json", "w") as f:
        json.dump(all_jobs, f, indent=2)

    # CSV (easy to inspect in Excel/Sheets)
    df = pd.DataFrame(all_jobs)
    df.to_csv("raw_jobs.csv", index=False)

    print(f"Done! Saved {len(all_jobs)} jobs to raw_jobs.csv and raw_jobs.json")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3))

    return df


if __name__ == "__main__":
    df = run_scraper(max_jobs=200)
