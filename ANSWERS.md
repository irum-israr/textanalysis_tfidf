Challenge 1: Conversational Filler & Numeric Noise
How the cluster top words change:

Before: Top words per cluster included conversational filler such as know, like, just, use, does, thanks and stray tokens/numbers (e.g., Cluster 0: know, like, don, just, does, use, thanks, good).

After: The top words become topical, e.g., mac, apple, drive, monitor, card, scsi, system (Mac hardware), and car, cars, engine, new, oil, ford (autos). Clusters become much easier to name and interpret.

Challenge 2: Hardware Vocabulary Overlap
Crosstab (pd.crosstab(df['true_label'], df['cluster'])):

Plaintext
cluster                  0   1    2    3
true_label                              
comp.sys.mac.hardware  269   0    2  639
rec.autos              396   0  507    2
sci.electronics        860   0   20   60
sci.med                875  71    0    2
Most overlapping categories: sci.electronics and comp.sys.mac.hardware. Many Mac-hardware posts land in cluster 0 (electronics).

Shared technical words: ['thanks', 'know', 'print', 'does', 'just', 'problem', 'don', 'like', 'use', 'want', 'work', 'info', 'new', 'vv', 'edu'] (along with overlapping physical terms like drive, card, board, and power).

How bigrams help: With ngram_range=(1, 2), ambiguous single words become distinctive phrases (e.g., hard drive, power mac, logic board vs. circuit board, power supply), separating the vocabularies using word context.

Challenge 3: Rant Length Bias (BoW vs. TF-IDF)Post∥x∥2​ in Xbow​∥x∥2​ in Xtfidf​Longest (#310, 4483 tokens)276.241.0000Shortest (#210, 1 tokens)1.001.0000

Why K-Means fails on raw BoW counts: K-Means relies on Euclidean distance. Long rants have massive vector norms while short posts have tiny norms, causing K-Means to cluster primarily by document length rather than actual topic.How TfidfVectorizer fixes it: It applies L2 normalization (norm='l2') by default, setting every document vector to a length of $1$ on the unit sphere. Distances then depend strictly on relative word importance (topic direction) rather than document magnitude (length).

Challenge 4: Regulated Compliance Explainability
explain_document(doc_id=412, top_n=5) output:

True Label: rec.autos | Assigned Cluster: 0

Top 5 terms and weights:

discussion — TF-IDF = 0.4524

feel — TF-IDF = 0.3989

days — TF-IDF = 0.3855

need — TF-IDF = 0.2885

meant — TF-IDF = 0.2667

Justification:
TF-IDF is fully transparent because every routing decision can be directly traced to specific words and numeric weights that human auditors can easily reproduce and verify. In contrast, black-box large language models cannot provide exact, deterministic, or auditable explanations for their outputs, posing severe compliance risks in regulated sectors like healthcare. Additionally, running TF-IDF and K-Means locally is computationally inexpensive and keeps sensitive patient data secure without relying on third-party APIs.
