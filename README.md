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

This is a system for performing **R**etrieval **A**ugmented **G**eneration. This system utilizes a hybrid search mechanism that employs the usage of lexical, bi-encoder, and cross-encoder rerankers that can be combined in sequence or parallel to perform passage retrieval over a large corpus conditioned on a query. Ther retrieved passages are then used for in-context learning with a **L**arge **L**anguage **M**odel to answer the query.

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

By selecting (1), you will be prompted to build a corpus for any directory that is in the `documents` directory within the project directory. If the corpus has already been built, you can instead use (2) to load a corpus from a directory in the `.corpus` directory in the project directory. If multiple corpi have been loaded, you can freely switch between them with (3), though this is not recommended for large corpi on machines with memory constraints. With (4), assuming a corpus has been loaded/selected, you may begin asking questions. Here is an example output:

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

The utilized GPU was a NVIDIA RTX 3070 (8GB VRAM) via CUDA on WSL. However, preprocessing for large corpi used in testing was done using GCP resources (NVIDIA T4 with 32 vCPUs and 108 GB RAM) in order to deal with memory-intensive parallel tasks such as processing text with SpaCy and encoding the passages.

## How it works

### Preprocessing

Preprocessing is done in multiple stages. First, the raw text of the documents is read, parsed, and cleaned. Following this, they are processed using SpaCy (`en_core_web_lg`). The processed SpaCy documents are then used to create a list of lists of keywords for lexical searches. This is done by removing stop words, punctuation, and whitespace while also reducing each term into their lemmas. This is used to both build a vocabulary of the unique lemmas and their embeddings present in the corpus as well as a BM25Plus object. During this, many IDs are also tracked in order to map indices between documents, passages, and sentences. from there passage embeddings are constructed using an embedding model.

### Retrieval

Retrieval is done with what is called a `RetrieverChain`, which is essentially a series of retrievers. It is constructed from a configuration dictionary that is essentially a list of configurations for the retrievers you wish to use. All retrievers use `k` as an input parameter. This determines how many retrieval results there will be. Typically, `k` will just be a natural number. However, there is experimental support for adaptive `k` retrieval. If set to -1, then the retrieval stops when the derivative of the sorted scores is at the minimum. This indicates the point of steepest descent. Alternatively, -2 instead retrieves up to the maximum acceleration in the scores, which is meant to capture saturation points. This doesn't tend to work, so it is not recommended. There is an issue in which it is desirable to choost a monotonic decreasing interpolator, but the PCHIP interpolatore is not necessarily stable for the second derivative. Furthermore, if the scores start rather saturated, then not many passages are going to be retrieved. Additionally for `k` less than one but greater than zero, one can set constant thresholds [0, 1]. This is only recommended for the cross-encoder and pooling retrievals, but can also be applied to the vector retrieval. This should not be applied to the BM25 retrieval, but it can be used for keyword expansion. This is meant to provide an option for a more specifically controllable thresholding, but it can still be rather corpus/query-dependent since it's hard to know what the distribution of scores will be a priori. For keyword expansion, if `k` is set to -3, then a special keyword expansion is performed. Using this method, if there are any query keywords with SpaCy vectors that are not present in the corpus, then the closest word in the corpus to that keyword is added to the query tokens. This tends to improve results. The reranker options are as follows:

#### BM25

This takes in the query, breaks it into lemmas of keywords (does not include stop words, punctuation, or whitespace), expands the keywords if specified, and then retrieves matching documents using the [BM25+](https://pypi.org/project/rank-bm25/) approach. This is a term-based takes in term frequencies and lengths to score passage relevance to a a given query. Term frequency (TF) generally tracks how many times a particular term appears in a document, though BM25 aqlos takes into account saturation in order to prevent placing greater emphasis on heavily repeated terms. Additionally, inverse document frequency tracks the importance of a term to the corpus as a whole, assigning higher weights to less frequent words and lower weights to less frequence words.  Additionally, there is a normalization component to take into account the fact that longer documents are likely to have more term occurences. The + variant is used in this system as a correction to address issues where the frequency normalization by document length is not lower-bounded, which can lead to situations where shorter documents with no relevant terms are scored similarly to longer documents that do have relevant terms. The SpaCy tokenizer was used to isolate terms. Stop words, punctuation, and whitespace were removed. Terms were also all lowercased and lemmatized in order to generally prevent sparsity issues in the vocabulary, though this can dilute information in some cases. The configuration looks like this:

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

This simply retrieves the top-k passages via cosine similarity between query and passage vectors (bi-encoding). The currently supported models are [LLM-Embedder](https://huggingface.co/BAAI/llm-embedder), [BGE-Large-EN-V1.5](https://huggingface.co/BAAI/bge-large-en-v1.5), and the [Instructor](https://huggingface.co/hkunlp/instructor-xl) family of embedding models. The emebedding models are used to encode queries and passages into vectors that capture the semantic meanings of the text. This can allow for matching deeper meanings of text that may elude lexical search methods. The supported models are instruction-based, which means that an instruction is prepended to the text that is embedded in order to shift the semantic features. This makes it so that you can encode a query such that its embedding is similar to passages containing information relevant to the queries. Similarity is measured via cosine similarity. All vectors are L2-normalized, so the cosine similarity is calculated via the inner product ($\cos\theta = \frac{\mathbf{u}\cdot\mathbf{v}}{||\mathbf{u}||||\mathbf{v}||} = \mathbf{u}\cdot\mathbf{v}$ for normalized vectors).

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

This uses a cross-encoder to rerank sentences by their relevance to a query directly. This is a more expensive reranker. If `passage_search` is set to `True`, then the results will be given for the top-k passages. Otherwise, it will be the top-k sentences (which at most corresponds to `k` passages, but can be less). The `pooling` keyword controls how the sentence scores will be pooled if the results are converted from sentence-wise ranking to passage-wise ranking. As opposed to a bi-encoder that uses embeddings that are separately encoded and then compared by a distance measure, a cross-encoder directly takes in two sequences and provides a classification according to the training objective (e.g. query relevance). Generally, cross-encoders have been noted to achieve better performance than bi-encoders, but they are much more computationally expensive since the calculations are pair-wise rather than independent. The supported cross-encoders are [MS-MARCO-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) and [BGE-Reranker-Large](https://huggingface.co/BAAI/bge-reranker-large).

```
{
    "name": "cross_encoder",
    "parameters": {
        "passage_search": True,
        "pooling": "arithmetic_mean",
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

This is a special reranker used to do mean-pooling of multiple rerankers in parallel. This reranker has its own `k` parameter and a `pooling` parameter for controlling how the scores will be combined (either max or a Pythagorean mean), but the additional parameters are simply a list of configurations for other rerankers, albeit with an additional `weight` field that determines how much they contribute to the final result. For right now, the scores from each reranker are independently normalized in order to ensure that they are comparable in scale. However, this may result in poor results in cases where one reranker has very little variance in scores (compared to the complete distribution) and another one doesn't. Here is an example configuration that uses an equally weighted BM25+ and vector similarity score:

```
{
    "name": "pool",
    "parameters": {
        "pooling": "arithmetic_mean",
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

Here you can see that the two rerankers are retrieving the top 1k results and then the top 100 passages from the arithmetic mean-pooled scores are retrieved. Passages that are retrieved by only some rerankers will diluted.

### Retrieval Performance

As it turns out, pooling the results of lexical and bi-encoder searches results in rather competitive performance. Here are the results over a subset of the BeIR information retrieval benchmark using the following retrieval configuration:

```
{
    "name": "pool",
    "parameters": {
        "pooling": "arithmetic_mean",
        "k": 100,
        "retriever_config": [
            {
                "name": "bm25",
                "parameters": {
                    "keyword_k": -3,
                    "k": 1000,
                },
                "weight": 0.25,
            },
            {
                "name": "vector",
                "parameters": {
                    "k": 1000,
                },
                "weight": 0.75,
            },
        ],
    },
}
```

Here are the results over five datasets in the BeIR benchmark (more to be added once I have a chance to run benchmarks).

|            | Lexical |            | Dense |              |      | Reranker |      |        |      |          |      |      | Pooling            |
| ---------- | :------: | :--------: | :---: | :----------: | :---: | :------: | :---: | :----: | :---: | :-------: | :---: | :---: | ------------------ |
|            |   BM25   |   BM25+   |  GTR  | LLM-Embedder |  BGE  |  monoT5  |      | RankT5 |      | RankLLaMA |      | SGPT | RAGMAN (BM25, BGE) |
|            | Standard | Lemmatized | 4.8B |     110M     | 335M |   220M   |  3B  |  220M  |  3B  |    7B    |  13B  | 5.8B | Lemmatized, 335M   |
| TREC-COVID |  0.656  |   0.620   | 0.501 |    0.776    | 0.763 |  0.778  | 0.795 | 0.790 | 0.824 |   0.852   | 0.861 | 0.873 | 0.774              |
| NFCorpus   |  0.325  |   0.320   | 0.342 |    0.362    | 0.371 |  0.357  | 0.384 | 0.373 | 0.399 |   0.303   | 0.284 | 0.363 | 0.386              |
| FiQA       |  0.236  |   0.234   | 0.467 |    0.371    | 0.450 |  0.414  | 0.514 | 0.413 | 0.493 |   0.465   | 0.481 | 0.372 | 0.461              |
| SCIDOCS    |  0.158  |   0.143   | 0.161 |    0.194    | 0.214 |  0.165  | 0.197 | 0.176 | 0.192 |   0.178   | 0.191 | 0.197 | 0.216              |
| SciFact    |  0.665  |   0.660   | 0.662 |    0.724    | 0.751 |  0.736  | 0.777 | 0.749 | 0.760 |   0.732   | 0.730 | 0.747 | 0.759              |
| Mean       |  0.408  |   0.395   | 0.427 |    0.485    | 0.510 |  0.490  | 0.533 | 0.500 | 0.534 |   0.506   | 0.509 | 0.510 | 0.518              |

As you can see, through the simple use of the pooling re-ranker with the custom BM25+ model and the BGE embedding model, reasonably competitive performance can be achieved. This implies that the approach can be used as a more economical alternative to using large models for retrieval, especially considering that the LLM used for question-answering will usually be rather large. The only two models that exceed the performance of the pooled BM25+ and BGE retrieval were the two 3B parameter T5-based rerankers. Notably, the performance of the lemmatized BM25+ model is worse than what has been reported in other works. This is likely due to the use of the SpaCy tokenizer, as performance is even worse when not performing any text cleaning, lowercasing, lemmatization, or keyword expansion.

### Answering

This tends to be the most resource-hungry component at runtime compared to the retrieval due to the use of an LLM. This essentially comes down to a two-step process in which the retrieved contexts are first split up such that the sentences are binned to fit within the given context window. In order to achieve this, the contexts are broken into sentences and the maximum number of sentences that can be fit into the maximum context token limit (maximum token limit - number of prompted query tokens) are grouped into bins. They must be contiguous. The passages must also appear in order (within each document). Higher scoring documents appear earlier in the contexts. For each binned context, an answer is generation via causal token generation. In the event that there are multiple contexts and thus multiple answers, an additional causal generation step is used to aggregate the answers into a single comprehensive answer. In order to reduce the memory imprint of the models, quantized variants are used. The currently supported models are [Llama-2-7B-Chat](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ) and [Mistral-7B-OpenOrca](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ), both with the default (main) GPTQ permutations. Both are instruction-tuned and optimized for chat dialogue use cases. The generation configurations are set to be deterministic (within machine precision anyway) by disabling multinomial sampling and using one beam. The number of maximum new tokens is configurable.

#### In-context

The first step in the question-answering component is the generation of in-context answers. The retrieved contexts are ordered such that the primary key is the relevance of the most relevant passage for each document and then the chronological ordering of the passages within each document as the secondary key. The maximum context length is determined by the difference between the maximum number of tokens supported by the model and the sum of the number of tokens in the system prompt and the maximum number of generated tokens. The contexts are broken into sentences and then binned (while retaining the order) to fit into the maximum context length. As such, for long retrieved contexts, an answer is generated over each context. The system prompt is as follows:

```
"You are a highly intelligence and accurate context-based question-answering assistant. You provide accurate and helpful responses to user queries based on the context provided by a context retrieval subsystem. Focus on directly answering the question instead of providing conversational responses. Strive to understand and respond to every question, even if it is not perfectly formed or clear. If a question is truly unanswerable due to factual inaccuracies or incoherence, gently clarify these issues with the user. If a question is beyond your knowledge base, be honest about your limitations, but refrain from sharing false or misleading information."
```

The prompt is designed to push the model towards providing direct "factual" answers based on provided contexts without embellishment.

#### Aggregation

In cases where there are multiple answers due to the context binning, an additional aggregation step is employed. In this case, the original query and then each of the generated answers are provided and the model is instructed to consolidate the different answers into a single comprehensive answer. This is a rather dirty solution to the problem since the contexts are no longer present, but it does allow for handling very long contexts through a two-step process of in-context answering and then across-context answer aggregation. The system prompt is as follows:

```
"You are a highly intelligent and accurate answer-aggregation assistant. You will be provided with a question and an enumerated list of answers that were generated from contexts provided by a context retrieval subsystem. Consolidate the answers into a single comprehensive and consistent answer that accurately addresses the question. Do not cite the original answers, only provide the consolidated answer. Focus on directly answering the question instead of providing conversational responses. Strive to understand and respond to every question, even if it is not perfectly formed or clear. If a question is truly unanswerable due to factual inaccuracies or incoherence, gently clarify these issues with the user. If a question is beyond your knowledge base, be honest about your limitations, but refrain from sharing false or misleading information.",
```

The prompt is designed similarly to the in-context answering prompt, albeit with a different description of the task in order to reflect that this is an aggregation rather than in-context answering task. The model is instructed to not cite the answers directly, but I have noticed there is still a tendency for models to use language like "From source 1", which doesn't really make much sense from the user perspective (the user does not see the binned contexts, just the original retrieved passages). Handling long contexts should be handled in a more elegant fashion in the future, but this is the approach for now.
