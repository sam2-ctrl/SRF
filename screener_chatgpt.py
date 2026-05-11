#!/usr/bin/env python3
"""Fetch company details + quarterly results from Screener and summarize with OpenAI."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

BASE_URL = "https://www.screener.in"


@dataclass
class CompanyData:
    name: str
    url: str
    about: str
    key_metrics: Dict[str, str]
    quarterly_results: List[Dict[str, str]]


def slugify_company(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug


def fetch_company_page(company_name: str) -> tuple[str, str]:
    """Try common Screener company URL patterns and return html + url."""
    candidates = [
        f"{BASE_URL}/company/{slugify_company(company_name)}/",
        f"{BASE_URL}/company/{company_name.upper().replace(' ', '')}/",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SRF-CompanyBot/1.0)",
    }

    for url in candidates:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200 and "Page not found" not in resp.text:
            return resp.text, url

    raise ValueError(
        f"Could not find company page for '{company_name}'. Try the exact Screener company code."
    )


def parse_company_data(html: str, url: str) -> CompanyData:
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.select_one("h1")
    name = title_tag.get_text(strip=True) if title_tag else "Unknown"

    about_tag = soup.select_one("div.company-profile p")
    about = about_tag.get_text(" ", strip=True) if about_tag else "Not available"

    key_metrics: Dict[str, str] = {}
    for ratio in soup.select("ul#top-ratios li"):
        label = ratio.select_one("span.name")
        value = ratio.select_one("span.number")
        if label and value:
            key_metrics[label.get_text(strip=True)] = value.get_text(" ", strip=True)

    quarterly_results: List[Dict[str, str]] = []
    q_table = soup.select_one('section#quarters table')
    if q_table:
        headers = [th.get_text(" ", strip=True) for th in q_table.select("thead th")]
        for row in q_table.select("tbody tr"):
            cells = [td.get_text(" ", strip=True) for td in row.select("td")]
            if len(cells) == len(headers):
                quarterly_results.append(dict(zip(headers, cells)))

    return CompanyData(
        name=name,
        url=url,
        about=about,
        key_metrics=key_metrics,
        quarterly_results=quarterly_results,
    )


def summarize_with_chatgpt(company_data: CompanyData, model: str = "gpt-4.1-mini") -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = f"""
You are a financial assistant.
Given Screener company data, provide:
1) A concise company overview
2) Important metrics in bullet points
3) A plain-English quarterly trend summary
4) 3 risk factors to watch

Company: {company_data.name}
Source URL: {company_data.url}
About: {company_data.about}
Key Metrics: {company_data.key_metrics}
Quarterly Results: {company_data.quarterly_results}
"""

    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )

    return resp.output_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Type a company name, fetch Screener data, and summarize with ChatGPT."
    )
    parser.add_argument("company", help="Company name or Screener code (e.g., TCS)")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument(
        "--raw", action="store_true", help="Print parsed raw data instead of GPT summary"
    )
    args = parser.parse_args()

    html, url = fetch_company_page(args.company)
    data = parse_company_data(html, url)

    if args.raw:
        print(data)
        return

    summary = summarize_with_chatgpt(data, model=args.model)
    print(summary)


if __name__ == "__main__":
    main()
