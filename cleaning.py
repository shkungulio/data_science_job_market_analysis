"""
cleaning.py
-----------
Takes the raw scraped data (raw_jobs.csv) and cleans it into a structured,
analysis-ready dataset (cleaned_jobs.csv).

This is the CORE of the project — every cleaning decision is documented
so you can explain it in interviews.

Run AFTER scraper.py.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime

# ── Skills dictionary ──────────────────────────────────────────────────────────
# We'll search for these keywords in the job description text.
# This is intentionally broad; add more as you spot them in the raw data.

SKILLS = {
    # Languages
    "python": ["python"],
    "r_language": [r"\bR\b", "r programming"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "bigquery"],
    "scala": ["scala"],
    "java": [r"\bjava\b"],
    "julia": ["julia"],

    # ML / DL frameworks
    "scikit_learn": ["scikit-learn", "sklearn"],
    "tensorflow": ["tensorflow", "tf2"],
    "pytorch": ["pytorch", "torch"],
    "keras": ["keras"],
    "xgboost": ["xgboost", "xgb"],
    "lightgbm": ["lightgbm", "lgbm"],

    # Data tools
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "spark": ["spark", "pyspark"],
    "dbt": [r"\bdbt\b"],
    "airflow": ["airflow"],

    # Visualization
    "tableau": ["tableau"],
    "powerbi": ["power bi", "powerbi"],
    "matplotlib": ["matplotlib"],
    "plotly": ["plotly"],

    # Cloud
    "aws": [r"\baws\b", "amazon web services", "sagemaker", "s3", "ec2"],
    "gcp": [r"\bgcp\b", "google cloud", "bigquery", "vertex ai"],
    "azure": [r"\bazure\b", "microsoft azure"],

    # MLOps / Deployment
    "docker": ["docker"],
    "kubernetes": ["kubernetes", r"\bk8s\b"],
    "mlflow": ["mlflow"],
    "git": [r"\bgit\b", "github", "gitlab"],

    # Techniques
    "nlp": [r"\bnlp\b", "natural language processing", "transformers", "bert", "llm"],
    "computer_vision": ["computer vision", r"\bcv\b", "image recognition"],
    "deep_learning": ["deep learning", "neural network"],
    "statistics": ["statistics", "statistical", "hypothesis testing", "bayesian"],
    "a_b_testing": ["a/b test", "a/b testing", "experimentation"],
}


# ── Cleaning Functions ─────────────────────────────────────────────────────────

def clean_title(title):
    """
    Normalize job titles. Handles None, extra whitespace, encoding issues.
    Does NOT try to standardize seniority here — that's a separate column.
    """
    if pd.isna(title):
        return None
    title = str(title).strip()
    # Remove common encoding artifacts
    title = title.replace("\u2019", "'").replace("\u2013", "-")
    # Collapse multiple spaces
    title = re.sub(r"\s+", " ", title)
    return title


def extract_seniority(title):
    """
    Extract seniority level from title text.
    Returns: 'junior', 'mid', 'senior', 'lead', 'manager', or 'unknown'
    """
    if pd.isna(title):
        return "unknown"
    title_lower = title.lower()

    if any(w in title_lower for w in ["junior", "jr.", "jr ", "entry", "associate"]):
        return "junior"
    elif any(w in title_lower for w in ["senior", "sr.", "sr ", "staff", "principal"]):
        return "senior"
    elif any(w in title_lower for w in ["lead", "head of", "director"]):
        return "lead"
    elif any(w in title_lower for w in ["manager", "vp ", "vice president"]):
        return "manager"
    elif any(w in title_lower for w in [" ii", " iii", " 2", " 3", "mid-level", "mid level"]):
        return "mid"
    else:
        return "unknown"


def clean_company(company):
    """Strip whitespace and fix encoding. Returns None for missing values."""
    if pd.isna(company):
        return None
    return str(company).strip()


def clean_location(location_raw):
    """
    Normalize location into: 'remote', 'usa', 'europe', 'worldwide', or
    the cleaned raw string if we can't categorize it.

    Decision log:
    - 'Anywhere' → 'worldwide'
    - 'USA Only', 'US Only' → 'usa_remote'
    - 'Europe Only' → 'europe_remote'
    - Blank/NaN → 'unknown'
    """
    if pd.isna(location_raw) or str(location_raw).strip() == "":
        return "unknown"

    loc = str(location_raw).strip().lower()

    if loc in ["anywhere", "worldwide", "global", "remote"]:
        return "worldwide"
    elif any(w in loc for w in ["usa only", "us only", "united states only", "north america"]):
        return "usa_remote"
    elif "europe" in loc:
        return "europe_remote"
    elif "canada" in loc:
        return "canada_remote"
    elif "uk" in loc or "united kingdom" in loc:
        return "uk_remote"
    else:
        # Return cleaned raw value — don't throw data away
        return str(location_raw).strip()


def parse_salary(salary_raw):
    """
    Convert raw salary strings to annual USD numeric values.
    Returns (salary_min, salary_max, salary_currency, salary_period) tuple.

    Decisions documented:
    - 'k' suffix → multiply by 1000
    - Hourly rates (detected by low magnitude) → multiply by 2080 (52 weeks × 40 hrs)
    - Ranges → extract min and max separately
    - Non-USD currencies flagged in salary_currency column
    - Ambiguous values → return NaN with note
    """
    if pd.isna(salary_raw) or str(salary_raw).strip() == "":
        return np.nan, np.nan, None, None

    raw = str(salary_raw).strip()

    # Detect currency
    if raw.startswith("£"):
        currency = "GBP"
    elif raw.startswith("€"):
        currency = "EUR"
    else:
        currency = "USD"

    # Remove currency symbols and commas
    cleaned = re.sub(r"[£€\$,]", "", raw)

    # Look for range pattern: e.g. "80k - 120k" or "80,000-120,000"
    range_match = re.search(r"([\d.]+)\s*k?\s*[-–]\s*([\d.]+)\s*k?", cleaned, re.IGNORECASE)
    single_match = re.search(r"([\d.]+)\s*k?", cleaned, re.IGNORECASE)

    def parse_number(s, has_k):
        n = float(s)
        if has_k or n < 1000:   # 'k' suffix or already in thousands
            n *= 1000
        return n

    if range_match:
        raw_min, raw_max = range_match.group(1), range_match.group(2)
        has_k = "k" in cleaned.lower()
        sal_min = parse_number(raw_min, has_k)
        sal_max = parse_number(raw_max, has_k)
    elif single_match:
        has_k = "k" in cleaned.lower()
        val = parse_number(single_match.group(1), has_k)
        sal_min = val
        sal_max = val
    else:
        return np.nan, np.nan, currency, None

    # Detect if hourly (values < 500 after parsing are almost certainly hourly)
    period = "annual"
    if sal_min < 500:
        sal_min *= 2080
        sal_max *= 2080
        period = "hourly_converted"

    return round(sal_min), round(sal_max), currency, period


def clean_date(date_raw):
    """
    Parse ISO date strings into a proper datetime.
    Returns None for unparseable values.
    """
    if pd.isna(date_raw):
        return None
    try:
        return pd.to_datetime(date_raw).date()
    except Exception:
        return None


def extract_skills(description):
    """
    Search the job description text for each skill in SKILLS dict.
    Returns a comma-separated string of matched skill names.

    Approach: regex matching on lowercase description, whole-word where possible.
    """
    if pd.isna(description):
        return ""

    text = str(description).lower()
    found = []

    for skill_name, patterns in SKILLS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(skill_name)
                break   # don't double-count same skill

    return ", ".join(found)


def is_duplicate(df):
    """
    Flag duplicates using: same company + similar title.
    Returns a boolean Series (True = duplicate, keep=False).

    Decision: we use exact match on (company, title) after lowercasing.
    A fuzzier approach (edit distance) would catch more, but for this
    project exact match is a good start — note this limitation in your writeup.
    """
    key = df["company_clean"].str.lower().fillna("") + "|" + df["title_clean"].str.lower().fillna("")
    return key.duplicated(keep="first")


# ── Main Cleaning Pipeline ─────────────────────────────────────────────────────

def run_cleaning(input_file="raw_jobs.csv", output_file="cleaned_jobs.csv"):
    """
    Full cleaning pipeline. Reads raw CSV, applies all cleaning steps,
    saves cleaned CSV with a cleaning report.
    """
    print(f"Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # ── Track cleaning stats ───────────────────────────────────────────────────
    report = {
        "raw_rows": len(df),
        "cleaning_steps": []
    }

    # ── Step 1: Clean titles ───────────────────────────────────────────────────
    print("Step 1: Cleaning titles...")
    df["title_clean"] = df["title"].apply(clean_title)
    df["seniority"] = df["title_clean"].apply(extract_seniority)
    seniority_dist = df["seniority"].value_counts().to_dict()
    report["cleaning_steps"].append({"step": "seniority_extraction", "distribution": seniority_dist})
    print(f"  Seniority distribution: {seniority_dist}")

    # ── Step 2: Clean companies ────────────────────────────────────────────────
    print("Step 2: Cleaning companies...")
    df["company_clean"] = df["company"].apply(clean_company)
    missing_company = df["company_clean"].isna().sum()
    report["cleaning_steps"].append({"step": "company_clean", "missing_count": int(missing_company)})
    print(f"  Missing company: {missing_company}")

    # ── Step 3: Deduplicate ────────────────────────────────────────────────────
    print("Step 3: Deduplicating...")
    df["is_duplicate"] = is_duplicate(df)
    n_dupes = df["is_duplicate"].sum()
    report["cleaning_steps"].append({"step": "deduplication", "duplicates_removed": int(n_dupes)})
    print(f"  Duplicates found: {n_dupes}")
    df = df[~df["is_duplicate"]].drop(columns=["is_duplicate"])

    # ── Step 4: Clean location ─────────────────────────────────────────────────
    print("Step 4: Cleaning location...")
    df["location_clean"] = df["location_raw"].apply(clean_location)
    loc_dist = df["location_clean"].value_counts().to_dict()
    report["cleaning_steps"].append({"step": "location_clean", "distribution": loc_dist})
    print(f"  Location distribution: {loc_dist}")

    # ── Step 5: Parse salary ───────────────────────────────────────────────────
    print("Step 5: Parsing salary...")
    salary_parsed = df["salary_raw"].apply(parse_salary)
    df["salary_min_usd"] = [x[0] for x in salary_parsed]
    df["salary_max_usd"] = [x[1] for x in salary_parsed]
    df["salary_currency"] = [x[2] for x in salary_parsed]
    df["salary_period"] = [x[3] for x in salary_parsed]
    df["salary_mid_usd"] = (df["salary_min_usd"] + df["salary_max_usd"]) / 2

    salary_found = df["salary_min_usd"].notna().sum()
    report["cleaning_steps"].append({
        "step": "salary_parsing",
        "jobs_with_salary": int(salary_found),
        "jobs_without_salary": int(len(df) - salary_found),
        "note": "Most jobs on WWR do not list salary — this is real-world data reality"
    })
    print(f"  Jobs with salary data: {salary_found} / {len(df)}")

    # ── Step 6: Parse dates ────────────────────────────────────────────────────
    print("Step 6: Parsing dates...")
    df["posted_date"] = df["posted_date_raw"].apply(clean_date)
    df["days_since_posted"] = (datetime.now().date() - df["posted_date"]).apply(
        lambda x: x.days if pd.notna(x) else None
    )
    report["cleaning_steps"].append({"step": "date_parsing", "missing_dates": int(df["posted_date"].isna().sum())})

    # ── Step 7: Extract skills from descriptions ───────────────────────────────
    print("Step 7: Extracting skills from descriptions...")
    df["skills_extracted"] = df["description"].apply(extract_skills)

    # Also create one boolean column per skill (useful for analysis)
    for skill in SKILLS.keys():
        df[f"skill_{skill}"] = df["skills_extracted"].str.contains(skill, na=False)

    top_skills = {
        skill: int(df[f"skill_{skill}"].sum())
        for skill in SKILLS.keys()
    }
    top_skills_sorted = dict(sorted(top_skills.items(), key=lambda x: x[1], reverse=True)[:10])
    report["cleaning_steps"].append({"step": "skill_extraction", "top_skills": top_skills_sorted})
    print(f"  Top skills found: {top_skills_sorted}")

    # ── Step 8: Final column selection ────────────────────────────────────────
    print("Step 8: Selecting final columns...")
    final_cols = [
        "title_clean", "company_clean", "seniority",
        "location_clean", "location_raw",
        "salary_min_usd", "salary_max_usd", "salary_mid_usd",
        "salary_currency", "salary_period", "salary_raw",
        "posted_date", "days_since_posted",
        "skills_extracted", "job_url", "scraped_at",
        "description",
    ] + [f"skill_{s}" for s in SKILLS.keys()]

    # Only keep columns that exist (in case some weren't scraped)
    final_cols = [c for c in final_cols if c in df.columns]
    df_clean = df[final_cols].copy()

    report["final_rows"] = len(df_clean)
    report["rows_removed_total"] = report["raw_rows"] - report["final_rows"]

    # ── Save outputs ───────────────────────────────────────────────────────────
    df_clean.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")
    print(f"Final shape: {df_clean.shape}")

    import json
    with open("cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Cleaning report saved to cleaning_report.json")

    return df_clean, report


if __name__ == "__main__":
    df_clean, report = run_cleaning()
    print("\n── Summary ──────────────────────────────────")
    print(f"Raw rows:      {report['raw_rows']}")
    print(f"Cleaned rows:  {report['final_rows']}")
    print(f"Removed:       {report['rows_removed_total']}")
