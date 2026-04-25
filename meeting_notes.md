# Meeting with Professor — DTU Researcher Finder
**Date: April 24, 2026**

---

## ENGLISH

### What I have built so far

The full end-to-end pipeline is implemented and working. Here is what each step does:

**1. Data acquisition (`src/acquire.py`)**
- Issues two SPARQL queries to Wikidata via the QLever endpoint
- First query: fetches paper titles and researcher names for all DTU employees (`wdt:P108 wd:Q1269766`)
- Second query: fetches topics (`wdt:P921`) for those same papers
- Saves everything to `data/raw_sparql.json`

**2. Preprocessing (`src/preprocess.py`)**
- Cleans and structures the raw data
- Deduplicates papers (one record per paper, with all linked researchers attached)
- Produces two text variants per paper:
  - `title_only` — just the paper title
  - `title_topics` — "Title: X  Topics: A, B, C"
- Output: `data/papers.json` and `data/researchers.json`

**3. Embedding (`src/embed.py`)**
- Uses `sentence-transformers` with model `all-MiniLM-L6-v2`
- Embeds all unique papers (not author-paper pairs) as L2-normalized float32 vectors
- Saves two matrices: `embeddings/embeddings_title_only.npy` and `embeddings/embeddings_title_topics.npy`
- Also saves `embeddings/paper_ids.json` to keep row order consistent

**4. Retrieval (`src/retrieve.py`)**
- `RetrievalIndex` class loads the embedding matrix once at startup
- Encodes the user query with the same model
- Computes dot product = cosine similarity (since all vectors are unit-normalized)
- Returns top-N most similar papers

**5. Aggregation (`src/aggregate.py`)**
- Collapses paper-level matches into a researcher-level ranking
- Three strategies implemented and comparable:
  - `max` — researcher score = best single paper score
  - `sum_top_3` — researcher score = sum of top-3 paper scores (default)
  - `mean_top_3` — researcher score = mean of top-3 paper scores
- Response includes matched publication titles and topics for each researcher

**6. REST API (`src/api.py`)**
Four endpoints, all working:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check service status, lists loaded indexes |
| `/version` | GET | Returns current version |
| `/v1/researcher-search` | POST | Main endpoint: query → ranked researchers |
| `/v1/debug-publication-search` | POST | Inspect paper-level scores (useful for debugging) |

The main endpoint accepts: `query`, `top_k`, `embedding_field` (`title_only` or `title_topics`), `aggregation` strategy, and `paper_pool` size.

**7. Evaluation (`evaluation/`)**
- `benchmark.json` — 6 hand-crafted queries with known expected researcher IDs
- `evaluate.py` — computes Hit@1, Hit@3, Hit@5, Hit@10, and MRR
- Can run `--compare-all` to benchmark all 6 field × strategy combinations at once

**8. Tests (`tests/`)**
- 7 unit tests covering aggregation logic and retrieval output properties
- `pytest tests/`

**9. Docker**
- `Dockerfile` and `docker-compose.yml` exist for containerized deployment

---

### What is NOT done yet

- The richer SPARQL query (professor offered to share one using both `affiliation` (`wdt:P1416`) and `employer` (`wdt:P108`) with an end-date filter to restrict to **current** DTU employees) — this has not been incorporated yet
- The evaluation has not been run to produce final numbers yet
- The embedding model used is `all-MiniLM-L6-v2` (open, free), not CampusAI embeddings — this is a design decision worth discussing

---

### Key things to know if asked

**Q: Why embed papers instead of researchers directly?**
A: A researcher's "research identity" is the union of their publications. By embedding each paper and then aggregating scores across papers per researcher, the system handles researchers with broad or narrow publication profiles uniformly.

**Q: Why two text variants?**
A: Topics from Wikidata (`wdt:P921`) are sparse — many papers have none. The `title_only` variant is the safe baseline. The `title_topics` variant enriches the representation when topics exist. Both are evaluated and compared.

**Q: Why three aggregation strategies?**
A: `max` rewards specialists (one very relevant paper). `sum_top_3` rewards researchers with multiple relevant papers (good for prolific researchers). `mean_top_3` is in between. Running the evaluation script tells us which works best empirically.

**Q: What embedding model did you use and why not CampusAI?**
A: `all-MiniLM-L6-v2` from sentence-transformers is open-source and runs locally. CampusAI embeddings could be swapped in — the pipeline is designed with that in mind (the embedding step is isolated in `embed.py`).

**Q: What about the SPARQL query improvement you mentioned?**
A: The current query only uses `wdt:P108` (employer), which includes former DTU employees. The professor mentioned a more general query using `affiliation` (`wdt:P1416`) and an end-date filter. This is the most important next step — ask him to share that query.

---
---

## ΕΛΛΗΝΙΚΑ

### Τι έχω υλοποιήσει μέχρι τώρα

Ολόκληρο το pipeline από την αρχή ως το τέλος είναι έτοιμο και λειτουργεί. Παρακάτω εξηγώ κάθε βήμα:

**1. Συλλογή δεδομένων (`src/acquire.py`)**
- Εκτελεί δύο ερωτήματα SPARQL στο Wikidata μέσω του QLever endpoint
- Πρώτο ερώτημα: παίρνει τίτλους εργασιών και ονόματα ερευνητών για όλους τους εργαζόμενους στο DTU (`wdt:P108 wd:Q1269766`)
- Δεύτερο ερώτημα: παίρνει τα θέματα (`wdt:P921`) για τις ίδιες εργασίες
- Αποθηκεύει τα πάντα στο `data/raw_sparql.json`

**2. Προεπεξεργασία (`src/preprocess.py`)**
- Καθαρίζει και δομεί τα ακατέργαστα δεδομένα
- Αφαιρεί διπλότυπα (ένα record ανά άρθρο, με όλους τους συνδεδεμένους ερευνητές)
- Φτιάχνει δύο παραλλαγές κειμένου ανά άρθρο:
  - `title_only` — μόνο ο τίτλος
  - `title_topics` — "Title: X  Topics: A, B, C"
- Έξοδος: `data/papers.json` και `data/researchers.json`

**3. Δημιουργία embeddings (`src/embed.py`)**
- Χρησιμοποιεί `sentence-transformers` με το μοντέλο `all-MiniLM-L6-v2`
- Κάνει embed κάθε μοναδικό άρθρο (όχι ζεύγη ερευνητή-άρθρου) ως L2-κανονικοποιημένα διανύσματα
- Αποθηκεύει δύο πίνακες: `embeddings_title_only.npy` και `embeddings_title_topics.npy`
- Επίσης αποθηκεύει `paper_ids.json` για να διατηρεί σωστή σειρά γραμμών

**4. Ανάκτηση (`src/retrieve.py`)**
- Η κλάση `RetrievalIndex` φορτώνει τον πίνακα embeddings μία φορά κατά την εκκίνηση
- Κωδικοποιεί το ερώτημα του χρήστη με το ίδιο μοντέλο
- Υπολογίζει εσωτερικό γινόμενο = cosine similarity (αφού τα διανύσματα είναι μοναδιαία)
- Επιστρέφει τα N πιο σχετικά άρθρα

**5. Συγκέντρωση αποτελεσμάτων (`src/aggregate.py`)**
- Μετατρέπει τα αποτελέσματα σε επίπεδο άρθρου σε κατάταξη σε επίπεδο ερευνητή
- Τρεις στρατηγικές για σύγκριση:
  - `max` — σκορ ερευνητή = καλύτερο σκορ ενός άρθρου
  - `sum_top_3` — σκορ ερευνητή = άθροισμα των top-3 σκορ (προεπιλογή)
  - `mean_top_3` — σκορ ερευνητή = μέσος όρος των top-3 σκορ
- Η απάντηση περιλαμβάνει τους τίτλους και τα θέματα των αντίστοιχων άρθρων

**6. REST API (`src/api.py`)**
Τέσσερα endpoints, όλα λειτουργικά:

| Endpoint | Μέθοδος | Σκοπός |
|---|---|---|
| `/health` | GET | Κατάσταση υπηρεσίας, ποια indexes είναι φορτωμένα |
| `/version` | GET | Τρέχουσα έκδοση |
| `/v1/researcher-search` | POST | Κύριο endpoint: ερώτημα → κατατεταγμένοι ερευνητές |
| `/v1/debug-publication-search` | POST | Βλέπει τα σκορ σε επίπεδο άρθρου (για debugging) |

Το κύριο endpoint δέχεται: `query`, `top_k`, `embedding_field`, στρατηγική `aggregation` και μέγεθος `paper_pool`.

**7. Αξιολόγηση (`evaluation/`)**
- `benchmark.json` — 6 ερωτήματα με γνωστά αναμενόμενα IDs ερευνητών
- `evaluate.py` — υπολογίζει Hit@1, Hit@3, Hit@5, Hit@10 και MRR
- Μπορεί να τρέξει `--compare-all` για σύγκριση όλων των συνδυασμών

**8. Tests (`tests/`)**
- 7 unit tests για τη λογική aggregation και τις ιδιότητες ανάκτησης
- `pytest tests/`

**9. Docker**
- `Dockerfile` και `docker-compose.yml` για containerized ανάπτυξη

---

### Τι δεν έχει γίνει ακόμα

- Το πιο γενικό SPARQL ερώτημα που πρόσφερε ο καθηγητής (χρήση και `affiliation` `wdt:P1416` και `employer` `wdt:P108` με φίλτρο end-date για **τρέχοντες** υπαλλήλους DTU) — δεν έχει ενσωματωθεί ακόμα
- Η αξιολόγηση δεν έχει τρέξει για να παράγει τελικούς αριθμούς
- Το μοντέλο embedding που χρησιμοποιείται είναι `all-MiniLM-L6-v2` (ανοιχτό, δωρεάν), όχι CampusAI embeddings — αυτό είναι απόφαση σχεδιασμού που αξίζει να συζητηθεί

---

### Σημαντικά πράγματα αν ρωτηθώ

**Γιατί κάνεις embed τα άρθρα και όχι απευθείας τους ερευνητές;**
Η "ερευνητική ταυτότητα" ενός ερευνητή είναι το σύνολο των δημοσιεύσεών του. Κάνοντας embed κάθε άρθρο και στη συνέχεια συγκεντρώνοντας τα σκορ ανά ερευνητή, το σύστημα χειρίζεται ομοιόμορφα και ερευνητές με ευρύ και με στενό δημοσιευτικό προφίλ.

**Γιατί δύο παραλλαγές κειμένου;**
Τα θέματα από το Wikidata (`wdt:P921`) είναι αραιά — πολλά άρθρα δεν έχουν. Το `title_only` είναι ασφαλής βάση σύγκρισης. Το `title_topics` εμπλουτίζει την αναπαράσταση όταν υπάρχουν θέματα. Και οι δύο αξιολογούνται και συγκρίνονται.

**Γιατί τρεις στρατηγικές aggregation;**
Το `max` ευνοεί ειδικούς (ένα πολύ σχετικό άρθρο). Το `sum_top_3` ευνοεί ερευνητές με πολλά σχετικά άρθρα. Το `mean_top_3` είναι ενδιάμεσο. Το evaluation script λέει ποιο δουλεύει καλύτερα εμπειρικά.

**Γιατί δεν χρησιμοποιείς CampusAI embeddings;**
Το `all-MiniLM-L6-v2` είναι open-source και τρέχει τοπικά. Τα CampusAI embeddings μπορούν να ενσωματωθούν — το pipeline το υποστηρίζει αυτό γιατί το βήμα embedding είναι απομονωμένο στο `embed.py`.

**Τι γίνεται με το SPARQL ερώτημα που ανέφερε ο καθηγητής;**
Το τρέχον ερώτημα χρησιμοποιεί μόνο `wdt:P108` (employer), που περιλαμβάνει και πρώην υπαλλήλους του DTU. Ο καθηγητής ανέφερε ένα πιο γενικό ερώτημα με `affiliation` (`wdt:P1416`) και φίλτρο end-date. Αυτό είναι το πιο σημαντικό επόμενο βήμα — να του ζητήσω να μου το μοιραστεί.
