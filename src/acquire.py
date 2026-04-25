"""
Fetch DTU researcher publication data from Wikidata via the QLever SPARQL endpoint.
Saves raw results to data/raw_sparql.json.

Run:
    python src/acquire.py
"""

import json
import time
import argparse
from pathlib import Path

import requests

QLEVER_ENDPOINT = "https://qlever.cs.uni-freiburg.de/api/wikidata"
DATA_DIR = Path(__file__).parent.parent / "data"

# Wikidata entity id for DTU. The queries below use it to restrict
# publications to researchers employed at DTU.
DTU_QID = "Q1269766"

# Fetch paper titles + researcher labels + optional topics.
# We run two queries and merge:
#   1. papers with titles
#   2. papers with topics (optional, may be sparse)
QUERY_TITLES = """
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
SELECT ?researcher ?researcherLabel ?paper ?paperTitle WHERE {
  ?researcher wdt:P108 wd:Q1269766 .
  ?paper      wdt:P50  ?researcher .
  ?paper      rdfs:label ?paperTitle .
  ?researcher rdfs:label ?researcherLabel .
  FILTER(LANG(?paperTitle)     = "en")
  FILTER(LANG(?researcherLabel) = "en")
}
"""

QUERY_TOPICS = """
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>
SELECT ?paper ?topic ?topicLabel WHERE {
  ?researcher wdt:P108 wd:Q1269766 .
  ?paper      wdt:P50  ?researcher .
  ?paper      wdt:P921 ?topic .
  ?topic      rdfs:label ?topicLabel .
  FILTER(LANG(?topicLabel) = "en")
}
"""


def sparql_query(query: str, endpoint: str = QLEVER_ENDPOINT) -> list[dict]:
    """Execute a SPARQL SELECT query; return list of binding dicts."""
    headers = {"Accept": "application/sparql-results+json"}
    params = {"query": query}
    resp = requests.get(endpoint, headers=headers, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    bindings = data["results"]["bindings"]
    return [
        {k: v["value"] for k, v in row.items()}
        for row in bindings
    ]


def strip_uri(uri: str) -> str:
    """Extract the QID or local name from a Wikidata URI."""
    return uri.rsplit("/", 1)[-1]


def fetch_all(endpoint: str = QLEVER_ENDPOINT) -> dict:
    # Titles and topics are fetched separately because topic coverage is sparse,
    # and keeping the queries independent makes the result merging simpler later.
    print("Querying paper titles ...")
    title_rows = sparql_query(QUERY_TITLES, endpoint)
    print(f"  -> {len(title_rows)} rows")

    print("Querying paper topics ...")
    time.sleep(1)  # be polite to the endpoint
    topic_rows = sparql_query(QUERY_TOPICS, endpoint)
    print(f"  -> {len(topic_rows)} rows")

    return {"title_rows": title_rows, "topic_rows": topic_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default=QLEVER_ENDPOINT,
        help="SPARQL endpoint URL",
    )
    parser.add_argument(
        "--out",
        default=str(DATA_DIR / "raw_sparql.json"),
        help="Output file path",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = fetch_all(args.endpoint)

    out_path = Path(args.out)
    # Persist the unmodified API payload so downstream steps can be rerun
    # without hitting the SPARQL endpoint again.
    out_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    print(f"Saved raw data to {out_path}")


if __name__ == "__main__":
    main()
