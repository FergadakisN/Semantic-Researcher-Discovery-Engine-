## Find DTU researcher given a text

Given a text, that could represent a (future) project title, an
article title or an project or article abstract find relevant DTU
researchers. This could be useful as a supervisor suggester.
It is envisioned that the publications of the researcher can be embedded
and the embeddings of all articles from DTU can be stored in a matrix.
The query string is compared with the embedding matrix and ordered
results are returns.
The publications of DTU researchers can be identified from one or more
SPARQL queries to triple stores that store Wikidata data. The articles
can be fetched and the title of the article is also available, - but not
the abstract. I have found around 50.000 articles-author pairs.
As part of the Web service you should return some information about
the DTU researcher.

A SPARQL query to get the topics of papers published by a DTU employee
is

```sparql
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?researcher ?paper ?topic WHERE {
  ?researcher wdt:P108 wd:Q1269766 .
  ?paper wdt:P50 ?researcher .
  ?paper wdt:P921 ?topic .
}
```

This can be issued against a QLever Wikidata endpoint https://qlever.dev/wikidata

With ChatGPT, I have vibe coded a Web app that does this relying on my
SPARQL query, the QLever endpoint and a dense vector representation via
CampusAI embedding. It works reasonably well.
