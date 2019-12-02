# openIEtriplesInference
Open IE-triples Inference Corpus
--- this corpus is related to LREC 2020 submission.

# TASK DEFINITION

**Open IE-triples are textual triples in the form of (Subj, Rel, Obj).**

Given a pair of open IE-triples $p$ (premise triple) and $h$ (hypothesis triple), the task of *open IE-triples inference* is to classify the relation between the $p$ and $h$ as:

* *Entailment:* if the meaning of $h$ can be inferred from the meaning of $p$,
* *Neutral:* if the assertion expressed by $h$ might be true in case of assertion expressed $p$ is true and, moreover, the case of entailment does not hold,
* *Contradiction:* if the meaning of $h$ is contradictory to the meaning of $p$.

Presented corpus is obtain by transformation process from SNLI and MultiNLI corpora (sentence pairs ID to SNLI and MultiNLI are provided).

Quantitative characteristics:
Train: 20234 instances
Dev: 2500 instances
Test: 2500 instances

Corpus provided in CSV format.

Columns
* GoldLabel
* Premise
* Hypothesis
* PairID
* PremiseSubj
* PremiseRel
* PremiseObj
* HypothesisSubj
* HypothesisRel
* HypothesisObj
