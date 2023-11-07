# RAGMAN

This is a system for doing **R**etrieval **A**ugmented **G**eneration. This system utilizes a hybrid search mechanism that employs the usage of lexical, bi-encoder, and cross-encoder rerankers that can be combined in sequence or parallel to perform passage retrieval over a large corpus conditioned on a query. Ther retrieved passages are then used for in-context learning with a **L**arge **L**anguage **M**odel to answer the query.

## Usage

In order to test the system out, I suggest using the `build_ir_documents.py` script in order to download an open retrieval dataset. Once they are downloaded, corpus pre-processing can be done with the `build_ir_corpi.py` and `build_ir_embeddings.py` scripts. For example:

```
python build_ir_documents.py --dataset nfcorpus
python build_ir_corpi.py --dataset nfcorpus
python build_ir_embeddings.py --dataset nfcorpus
```

From there, you can use the `test_ragman.py` script to try the system out. You can also use a custom document set composed of `.PDF` files, `.json` files (where each file is a list of passages as strings), or `.txt` files (where each line is a passage). Note that PDF processing is still a work in progress and may not always perform well with the reader that is employed. In order to build a custom corpus, you may also use the `test_ragman.py` script. Here is what the interface looks like upon launching:

```
=======================================================================================================================
Welcome to RAGMAN.
=======================================================================================================================
-----------------------------------------------------------------------------------------------------------------------
Please select an action:
1) Build corpus
2) Load corpus
3) Select corpus
4) Query corpus
Q) Exit
-----------------------------------------------------------------------------------------------------------------------
Selection:
```

By selecting (1), you will be prompted to build a corpus for any directory that is in the `documents` directory within the project directory. If the corpus has already been built, you can instead use (2) to load a corpus from a directory in the `.corpus` directory in the project directory. If multiple corpi have been loaded, you can freely switch between them with (3), though this is not recommended for large corpi on machines with memory constraints. With (4), assuming a corpus has been loaded/selected, you may being asking questions. Here is an example output:

```
-----------------------------------------------------------------------------------------------------------------------
Please select an action:
1) Build corpus
2) Load corpus
3) Select corpus
4) Query corpus
Q) Exit
-----------------------------------------------------------------------------------------------------------------------
Selection: 4
-----------------------------------------------------------------------------------------------------------------------
Selected corpus: nfcorpus
Please enter a query (B to go back and Q to quit): Do Cholesterol Statin Drugs Cause Breast Cancer?
-----------------------------------------------------------------------------------------------------------------------
Searching...
-----------------------------------------------------------------------------------------------------------------------
Retrieving with Retriever Chain
Vector: Ranking Passages [processing]
Generating Answers Over Contexts:   0%|                                                          | 0/1 [00:00<?, ?it/s]The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:32000 for open-end generation.
Generating Answers Over Contexts: 100%|██████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.17s/it]
-----------------------------------------------------------------------------------------------------------------------
Context:
Document: passages.json
Page: 1
Top Score: 0.9219433069229126
Recent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence but the effect on disease-specific mortality remains unclear. We evaluated risk of breast cancer death among statin users in a population-based cohort of breast cancer patients. The study cohort included all newly diagnosed breast cancer patients in Finland during 19952003 (31,236 cases), identified from the Finnish Cancer Registry. Information on statin use before and after the diagnosis was obtained from a national prescription database. We used the Cox proportional hazards regression method to estimate mortality among statin users with statin use as time-dependent variable. A total of 4,151 participants had used statins. During the median follow-up of 3.25 years after the diagnosis (range 0.089.0 years) 6,011 participants died, of which 3,619 (60.2%) was due to breast cancer. After adjustment for age, tumor characteristics, and treatment selection, both post-diagnostic and pre-diagnostic statin use were associated with lowered risk of breast cancer death (HR 0.46, 95% CI 0.380.55 and HR 0.54, 95% CI 0.440.67, respectively). The risk decrease by post-diagnostic statin use was likely affected by healthy adherer bias; that is, the greater likelihood of dying cancer patients to discontinue statin use as the association was not clearly dose-dependent and observed already at low-dose/short-term use. The dose- and time-dependence of the survival benefit among pre-diagnostic statin users suggests a possible causal effect that should be evaluated further in a clinical trial testing statins' effect on survival in breast cancer patients.

Document: passages.json
Page: 2
Top Score: 0.9157405495643616
BACKGROUND: Preclinical studies have shown that statins, particularly simvastatin, can prevent growth in breast cancer cell lines and animal models. We investigated whether statins used after breast cancer diagnosis reduced the risk of breast cancer-specific, or all-cause, mortality in a large cohort of breast cancer patients. METHODS: A cohort of 17,880 breast cancer patients, newly diagnosed between 1998 and 2009, was identified from English cancer registries (from the National Cancer Data Repository). This cohort was linked to the UK Clinical Practice Research Datalink, providing prescription records, and to the Office of National Statistics mortality data (up to 2013), identifying 3694 deaths, including 1469 deaths attributable to breast cancer. Unadjusted and adjusted hazard ratios (HRs) for breast cancer-specific, and all-cause, mortality in statin users after breast cancer diagnosis were calculated using time-dependent Cox regression models. Sensitivity analyses were conducted using multiple imputation methods, propensity score methods and a case-control approach. RESULTS: There was some evidence that statin use after a diagnosis of breast cancer had reduced mortality due to breast cancer and all causes (fully adjusted HR = 0.84 [95% confidence interval = 0.68-1.04] and 0.84 [0.72-0.97], respectively). These associations were more marked for simvastatin 0.79 (0.63-1.00) and 0.81 (0.70-0.95), respectively. CONCLUSIONS: In this large population-based breast cancer cohort, there was some evidence of reduced mortality in statin users after breast cancer diagnosis. However, these associations were weak in magnitude and were attenuated in some sensitivity analyses.

Document: passages.json
Page: 1373
Top Score: 0.863933801651001
This paper is based on a longer report on the benefits, safety and modalities of information representation with regard to women and statin use, situated within the historical context of Women's Health Movement which has advocated for unbiased, appropriate medical research and prescribing for women based on the goals of full-disclosure, informed consent, evidence-based medicine and gender-based analysis. The evidence base for prescribing statins for women, especially for primary prevention is weak, yet Canadian data suggest that half of all prescriptions are for women. Safety meta-analyses do not disaggregate for women; do not consider female vulnerability to statin induced muscle problems, and women-centred concerns such as breast-cancer, miscarriage or birth defects are under-researched. Many trials have not published their non-cardiac serious adverse event data. These factors suggest that the standards of full-disclosure, informed consent, evidence-based prescribing and gender-based analysis are not being met and women should proceed with caution.

Document: passages.json
Page: 1374
Top Score: 0.9176362752914429
Emerging evidence suggests that statins' may decrease the risk of cancers. However, available evidence on breast cancer is conflicting. We, therefore, examined the association between statin use and risk of breast cancer by conducting a detailed meta-analysis of all observational studies published regarding this subject. PubMed database and bibliographies of retrieved articles were searched for epidemiological studies published up to January 2012, investigating the relationship between statin use and breast cancer. Before meta-analysis, the studies were evaluated for publication bias and heterogeneity. Combined relative risk (RR) and 95 % confidence interval (CI) were calculated using a random-effects model (DerSimonian and Laird method). Subgroup analyses, sensitivity analysis, and cumulative meta-analysis were also performed. A total of 24 (13 cohort and 11 case-control) studies involving more than 2.4 million participants, including 76,759 breast cancer cases contributed to this analysis. We found no evidence of publication bias and evidence of heterogeneity among the studies. Statin use and long-term statin use did not significantly affect breast cancer risk (RR = 0.99, 95 % CI = 0.94, 1.04 and RR = 1.03, 95 % CI = 0.96, 1.11, respectively). When the analysis was stratified into subgroups, there was no evidence that study design substantially influenced the effect estimate. Sensitivity analysis confirmed the stability of our results. Cumulative meta-analysis showed a change in trend of reporting risk of breast cancer from positive to negative in statin users between 1993 and 2011. Our meta-analysis findings do not support the hypothesis that statins' have a protective effect against breast cancer. More randomized clinical trials and observational studies are needed to confirm this association with underlying biological mechanisms in the future.

Document: passages.json
Page: 1376
Top Score: 0.9045752286911011
Background Mechanistic studies largely support the chemopreventive potential of statins. However, results of epidemiologic studies investigating statin use and breast cancer risk have been inconsistent and lacked the ability to evaluate long-term statin use. Materials and Methods We utilized data from a population-based case-control study of breast cancer conducted in the Seattle-Puget Sound region to investigate the relationship between long-term statin use and breast cancer risk. 916 invasive ductal carcinoma (IDC) and 1,068 invasive lobular carcinoma (ILC) cases 55-74 years of age diagnosed between 2000 and 2008 were compared to 902 control women. All participants were interviewed in-person and data on hypercholesterolemia and all episodes of lipid lowering medication use were collected through a structured questionnaire. We assessed the relationship between statin use and IDC and ILC risk using polytomous logistic regression. Results Current users of statins for 10 years or longer had a 1.83-fold increased risk of IDC [95% confidence interval (CI): 1.14-2.93] and a 1.97-fold increased risk of ILC (95% CI: 1.25-3.12) compared to never users of statins. Among women diagnosed with hypercholesterolemia, current users of statins for 10 years or longer had more than double the risk of both IDC [odds ratio (OR): 2.04, 95% CI: 1.17-3.57] and ILC (OR: 2.43, 95% CI: 1.40-4.21) compared to never users. Conclusion In this contemporary population-based case-control study long-term use of statins was associated with increased risks of both IDC and ILC. Impact Additional studies with similarly high frequencies of statin use for various durations are needed to confirm this novel finding.

-----------------------------------------------------------------------------------------------------------------------
Query: do cholesterol statin drugs cause breast cancer?
Answer: The available evidence on the relationship between cholesterol statin drugs and breast cancer is conflicting and inconclusive. Some studies suggest that statins may have a protective effect against breast cancer, while others indicate that long-term use of statins is associated with an increased risk of breast cancer. More randomized clinical trials and observational studies are needed to confirm this association and understand the underlying biological mechanisms.
-----------------------------------------------------------------------------------------------------------------------
Selected corpus: nfcorpus
Please enter a query (B to go back and Q to quit):
```

The search mechanism used here was a simple vector search in which the top 5 passages (by cosine similarity between query and passage embedding) were chosen using the [LLM-Embedder model](https://huggingface.co/BAAI/llm-embedder). The scores are the associated cosine similarities. Note that the passages are given in the order in which they appear in the original corpus rather than by descending score. In practice, when dealing with multiple documents, the passages will be ordered with the score as the primary key and the passage index as the secondary key. That is to mean that passages will be ordered by highest scoring document and then by temporal order within each document. The meaning of the scores may vary depending on the search configuration.

## How it works

### Preprocessing

Preprocessing is done in multiple stages. First, the raw text of the documents is read, parsed, and cleaned. Following this, they are processed using SpaCy (`en_core_web_lg`). The processed SpaCy documents are then used to create a list of lists of keywords for lexical searches. This is done by removing stop words, punctuation, and whitespace while also reducing each term into their lemmas. This is used to both build a vocabulary of the unique lemmas and their embeddings present in the corpus as well as a BM25Plus object. During this, many IDs are also tracked in order to map indices between documents, passages, and sentences. from there passage embeddings are constructed using an embedding model.

### Retrieval

Retrieval is done with what is called a `RetrieverChain`, which is essentially a series of retrievers. It is constructed from a configuration dictionary that is essentially a list of configurations for the retrievers you wish to use. The options are as follows:

#### BM25

This takes in the query, breaks it into lemmas of keywords (does not include stop words, punctuation, or whitespace), expands the keywords if specified, and then retrieves matching documents using the BM25+ approach. The configuration looks like this:

```
{
    "name": "bm25",
    "parameters": {
        "keyword_k": 0,
        "k": 100,
    },
}      
```

Where the `name` field specifies the retriever and `parameters` specifies the search parameters. By changing `keyword_k`, you can instruct the retriever to grab the top-k similar keywords (by vector cosine similarity using SpaCy vectors) for each keyword extracted from the query. These will always be lemmas of words that appear explicitly in the corpus. And then `k` specifies the top-k passages to be retrieved. This is a passage-based reranker.

#### Vector

This simply retrieves the top-k passages via cosine similarity between query and passage vectors.

```
{
    "name": "vector",
    "parameters": {
        "k": 100,
    },
}
```

This is a passage-based reranker.

#### Cross-Encoder

This uses a cross-encoder to rerank sentences by their relevance to a query directly. This is a more expensive reranker. If `passage_search` is set to `True`, then the results will be given for the top-k passages. Otherwise, it will be the top-k sentences (which at most corresponds to `k` passages, but can be less).

```
{
    "name": "cross_encoder",
    "parameters": {
        "passage_search": True,
        "k": 100,
    },
},
```

This is a sentence-based reranker.

#### Graph

This is a graph-based reranking method that is best used after other rerankers (if at all). It essentially constructs a neighbor graph between passages and finds "hub" passages that are highly relevant to other passages in order to rerank results. It is not directly conditioned on the query, but if it follows a reranker that is, then you should still expect to get passages that are relevant to relevant passages. This is somewhat similar to PageRank or HITS, but it uses similarity scores instead of hyperlinks to establish linking between passages and supports negative relationships.
