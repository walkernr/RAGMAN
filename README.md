```
  o__ __o                o           o__ __o       o          o           o           o          o  
 <|     v\              <|>         /v     v\     <|\        /|>         <|>         <|\        <|> 
 / \     <\             / \        />       <\    / \\o    o// \         / \         / \\o      / \ 
 \o/     o/           o/   \o    o/               \o/ v\  /v \o/       o/   \o       \o/ v\     \o/ 
  |__  _<|           <|__ __|>  <|       _\__o__   |   <\/>   |       <|__ __|>       |   <\     |  
  |       \          /       \   \\          |    / \        / \      /       \      / \    \o  / \ 
 <o>       \o      o/         \o   \         /    \o/        \o/    o/         \o    \o/     v\ \o/ 
  |         v\    /v           v\   o       o      |          |    /v           v\    |       <\ |  
 / \         <\  />             <\  <\__ __/>     / \        / \  />             <\  / \        < \ 
                                                                                                  
                                                                                                  
```

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

The search mechanism used here was a simple vector search in which the top 5 passages (by cosine similarity between query and passage embedding) were chosen using the [LLM-Embedder model](https://huggingface.co/BAAI/llm-embedder). The scores are the associated cosine similarities. Note that the passages are given in the order in which they appear in the original corpus rather than by descending score. In practice, when dealing with multiple documents, the passages will be ordered with the score as the primary key and the passage index as the secondary key. That is to mean that passages will be ordered by highest scoring document and then by temporal order within each document. The meaning of the scores may vary depending on the search configuration. Note that while preprocessing works best with a rather powerful machine, the system can be deployed on a workstation. For instance, I used a machine with the following hardware/software for testing:

```
            .-/+oossssoo+/-.               ████████@█████████
        `:+ssssssssssssssssss+:`           ------------------
      -+ssssssssssssssssssyyssss+-         OS: Ubuntu 20.04.6 LTS on Windows 10 x86_64
    .ossssssssssssssssssdMMMNysssso.       Kernel: 5.15.90.1-microsoft-standard-WSL2
   /ssssssssssshdmmNNmmyNMMMMhssssss/      Uptime: 5 days, 46 mins
  +ssssssssshmydMMMMMMMNddddyssssssss+     Packages: 919 (dpkg)
 /sssssssshNMMMyhhyyyyhmNMMMNhssssssss/    Shell: zsh 5.8
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Theme: Adwaita [GTK3]
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Icons: Adwaita [GTK3]
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   Terminal: Relay(9)
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   CPU: AMD Ryzen 9 3950X (32) @ 3.493GHz
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   GPU: 8d05:00:00.0 Microsoft Corporation Device 008e
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Memory: 724MiB / 48173MiB
 /sssssssshNMMMyhhyyyyhdNMMMNhssssssss/
  +sssssssssdmydMMMMMMMMddddyssssssss+
   /ssssssssssshdmNNNNmyNMMMMhssssss/
    .ossssssssssssssssssdMMMNysssso.
      -+sssssssssssssssssyyyssss+-
        `:+ssssssssssssssssss+:`
            .-/+oossssoo+/-.
```

The GPU utilized was a NVIDIA RTX 3070 via CUDA on WSL. However, preprocessing for large corpi used in testing was done using GCP resources (NVIDIA T4 with 32 vCPUs and 108 GB RAM) in order to deal with memory-intensive parallel tasks such as processing text with SpaCy and encoding the passages.

## How it works

### Preprocessing

Preprocessing is done in multiple stages. First, the raw text of the documents is read, parsed, and cleaned. Following this, they are processed using SpaCy (`en_core_web_lg`). The processed SpaCy documents are then used to create a list of lists of keywords for lexical searches. This is done by removing stop words, punctuation, and whitespace while also reducing each term into their lemmas. This is used to both build a vocabulary of the unique lemmas and their embeddings present in the corpus as well as a BM25Plus object. During this, many IDs are also tracked in order to map indices between documents, passages, and sentences. from there passage embeddings are constructed using an embedding model.

### Retrieval

Retrieval is done with what is called a `RetrieverChain`, which is essentially a series of retrievers. It is constructed from a configuration dictionary that is essentially a list of configurations for the retrievers you wish to use. All retrievers use `k` as an input parameter. This determines how many retrieval results there will be. Typically, `k` will just be a natural number. However, there is experimental support for adaptive `k` retrieval. If set to -1, then the retrieval stops when the derivative of the sorted scores is at the minimum. This indicates the point of steepest descent. Alternatively, -2 instead retrieves up to the maximum acceleration in the scores, which is meant to capture saturation points. This doesn't tend to work, so it is not recommended. There is an issue in which it is desirable to choost a monotonic decreasing interpolator, but the PCHIP interpolatore is not necessarily stable for the second derivative. Furthermore, if the scores start rather saturated, then not many passages are going to be retrieved. Additionally for `k` less than one but greater than zero, the cross-encoder and pooling rerankers allow for setting constant thresholds since they are both bounded on [0, 1]. This is meant to provide an option for a more specifically controllable thresholding, but it can still be rather corpus/query-dependent since it's hard to know what the distribution of scores will be a priori. This is even worse with BM25 and embedding models, so this is not provided for those rerankers (though they may be later just for testing). The reranker options are as follows:

#### BM25

This takes in the query, breaks it into lemmas of keywords (does not include stop words, punctuation, or whitespace), expands the keywords if specified, and then retrieves matching documents using the BM25+ approach. This is a term-based takes in term frequencies and lengths to score passage relevance to a a given query. Term frequency (TF) generally tracks how many times a particular term appears in a document, though BM25 aqlos takes into account saturation in order to prevent placing greater emphasis on heavily repeated terms. Additionally, inverse document frequency tracks the importance of a term to the corpus as a whole, assigning higher weights to less frequent words and lower weights to less frequence words.  Additionally, there is a normalization component to take into account the fact that longer documents are likely to have more term occurences. The + variant is used in this system as a correction to address issues where the frequency normalization by document length is not lower-bounded, which can lead to situations where shorter documents with no relevant terms are scored similarly to longer documents that do have relevant terms. The SpaCy tokenizer was used to isolate terms. Stop words, punctuation, and whitespace were removed. Terms were also all lowercased and lemmatized in order to generally prevent sparsity issues in the vocabulary, though this can dilute information in some cases. The configuration looks like this:

```
{
    "name": "bm25",
    "parameters": {
        "keyword_k": 0,
        "k": 100,
    },
}      
```

Where the `name` field specifies the retriever and `parameters` specifies the search parameters. By changing `keyword_k`, you can instruct the retriever to grab the top-k similar keywords (by vector cosine similarity using SpaCy vectors) for each keyword extracted from the query. These will always be lemmas of words that appear explicitly in the corpus. And then `k` specifies the top-k passages to be retrieved. The query terms will always be the unique set of terms present in the query. This is a passage-based reranker.

#### Vector

This simply retrieves the top-k passages via cosine similarity between query and passage vectors (bi-encoding). The currently supported models are LLM-Embedder, BGE-Large-EN-V1.5, and the Instructor family of embedding models. The emebedding models are used to encode queries and passages into vectors that capture the semantic meanings of the text. This can allow for matching deeper meanings of text that may elude lexical search methods. The supported models are instruction-based, which means that an instruction is prepended to the text that is embedded in order to shift the semantic features. This makes it so that you can encode a query such that its embedding is similar to passages containing information relevant to the queries. Similarity is measured via cosine similarity. All vectors are L2-normalized, so the cosine similarity is calculated via the inner product ($\cos\theta = \frac{\bold{u}\cdot\bold{v}}{||\bold{u}||||\bold{v}||} = \bold{u}\cdot\bold{v}$ for normalized vectors).

```
{
    "name": "vector",
    "parameters": {
        "k": 100,
    },
}
```

This is a passage-based reranker, but a sentence-level version can be implemented. Earlier versions included this in preprocessing, but that became far too taxing on disk space to be reasonable for a larger corpus of tens of millions of sentences to be particularly reasonable. This can feasibly be done on the fly, though, as with the cross-encoder.

#### Cross-Encoder

This uses a cross-encoder to rerank sentences by their relevance to a query directly. This is a more expensive reranker. If `passage_search` is set to `True`, then the results will be given for the top-k passages. Otherwise, it will be the top-k sentences (which at most corresponds to `k` passages, but can be less). As opposed to a bi-encoder that uses embeddings that are separately encoded and then compared by a distance measure, a cross-encoder directly takes in two sequences and provides a classification according to the training objective (e.g. query relevance). Generally, cross-encoders have been noted to achieve better performance than bi-encoders, but they are much more computationally expensive since the calculations are pair-wise rather than independent.

```
{
    "name": "cross_encoder",
    "parameters": {
        "passage_search": True,
        "k": 100,
    },
},
```

This is a sentence-based reranker (passage-based may be supported in the future).

#### Graph

This is a graph-based reranking method that is best used after other rerankers (if at all). It essentially constructs a neighbor graph between passages and finds "hub" passages that are highly relevant to other passages in order to rerank results. It is not directly conditioned on the query, but if it follows a reranker that is, then you should still expect to get passages that are relevant to relevant passages. This is somewhat similar to PageRank or HITS, but it uses similarity scores instead of hyperlinks to establish linking between passages and supports negative relationships. The cosine similarity matrix is used to construct a weighted graph and the scores are initialized as the scores from the previous reranker output. Additional passages that were not provided by the previous reranker are included as long as their similarity score with a passage that was chosen is at least 0.90 and also in the 90th percentile of similarities between each previous passage hit and all passages. For right now, these thresholds are not controllable parameters, but they might be in the future.

```
{
    "name": "graph",
    "parameters": {
        "k": 100,
    },
}
```

This is a passage-based reranker.

#### Pool

This is a special reranker used to do mean-pooling of multiple rerankers in parallel. This reranker has its own `k` parameter, but the additional parameters are simply a list of configurations for other rerankers, albeit with an additional `weight` field that determines how much they contribute to the final result. For right now, the scores from each reranker are independently normalized in order to ensure that they are comparable in scale. However, this may result in poor results in cases where one reranker has very little variance in scores (compared to the complete distribution) and another one doesn't. Here is an example configuration that uses an equally weighted BM25+ and vector similarity score:

```
{
    "name": "pool",
    "parameters": {
        "k": 100,
        "retriever_config": [
            {
                "name": "bm25",
                "parameters": {
                    "keyword_k": 0,
                    "k": 1000,
                },
                "weight": 0.5,
            },
            {
                "name": "vector",
                "parameters": {
                    "k": 1000,
                },
                "weight": 0.5,
            },
        ],
    },
}
```

Here you can see that the two rerankers are retrieving the top 1k results and then the top 100 passages from the mean-pooled scores are retrieved. Passages that are retrieved by only some rerankers will diluted. Additional pooling options such as max pooling or other Pythagorean means may be supported in the future. This is a passage-based reranker.

### Answering


#### In-context


#### Aggregation
