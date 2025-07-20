<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Week 1 Foundations of Retrieval-Augmented Generation (RAG)

In Week 1 you gain the conceptual toolkit on which every later enhancement depends. This guide curates the most authoritative reading, viewing, and hands-on resources for five core pillars—(1) RAG architecture, (2) embeddings, (3) chunking \& text-splitting, (4) vector databases \& indexing, and (5) retrieval algorithms. Each section lists foundational papers or docs, explanatory blogs, and practical tutorials or notebooks so you can move seamlessly from theory to code.

![Diagram illustrating the three-step Retrieval Augmented Generation (RAG) pipeline: data ingestion into a vector database, retrieval of context based on user query, and generation of response by a language model.](https://pplx-res.cloudinary.com/image/upload/v1748549356/pplx_project_search_images/250787dc76d5b39f0927dd1628ffb702a8a74db8.jpg)

Diagram illustrating the three-step Retrieval Augmented Generation (RAG) pipeline: data ingestion into a vector database, retrieval of context based on user query, and generation of response by a language model.

## 1.  RAG Architecture \& End-to-End Pipeline

### 1.1  Seminal Papers

* Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks—Lewis et al., Meta AI (2020) [^1][^2]
* Retrieval-Augmented Generation for Large Language Models: A Survey (2023) [^3]


### 1.2  Cloud \& Vendor Guides

* Azure Architecture Center: Design \& Evaluate a RAG Solution [^4]
* IBM Architecture Pattern for RAG [^5]
* AWS “What is RAG?” explainer [^6]


### 1.3  Step-by-Step Tutorials

* LangChain RAG Tutorial Part 1—build an app from scratch [^7]
* Pinecone Learn: Retrieval-Augmented Generation series [^8]

**Watch**

* YouTube: “Optimize RAG with Hybrid Search \& Reranking” (code demo) [^9]


## 2.  Embeddings \& Representation Learning

![Vector database workflow illustrating the write and query paths, embedding model usage, indexing, and approximate nearest neighbor search in embedding space.](https://pplx-res.cloudinary.com/image/upload/v1749152986/pplx_project_search_images/366955a23fa74b2eb798d132e79c9cfe0bbe7d4c.jpg)

Vector database workflow illustrating the write and query paths, embedding model usage, indexing, and approximate nearest neighbor search in embedding space.

### 2.1  Choosing an Embedding Model

* MongoDB blog: How to choose the best embedding model for your LLM application [^10]
* Pinecone: Rundown of E5 vs Cohere v3 vs Ada 002 [^11]
* DataStax: 2025 benchmark of proprietary \& open-source models [^12]
* MTEB Leaderboard—interactive benchmark explorer [^13]


### 2.2  Generating \& Evaluating Embeddings

* Azure “Generate Embeddings” phase guide [^14]
* OpenAI knowledge-retrieval doc on chunk-level embeddings [^15]
* Faiss “Missing Manual” chapter on product quantization [^16]


### 2.3  Hands-On Notebooks

* Hugging Face transformers: DPR, Sentence-BERT, E5 quick-start [^17]
* GitHub nanoDPR: 300-line dense retriever training [^18]


## 3.  Chunking, Text-Splitting \& Context Windows

![Comparison of naive (i.i.d.) chunking versus late chunking approaches for embedding long documents, highlighting how late chunking preserves contextual information.](https://pplx-res.cloudinary.com/image/upload/v1748874798/pplx_project_search_images/8d68d27a349f52e9bd7282962bbfd536556824b6.jpg)

Comparison of naive (i.i.d.) chunking versus late chunking approaches for embedding long documents, highlighting how late chunking preserves contextual information.

### 3.1  Why Chunk Size Matters

* Unstructured.io best practices for chunking [^19]
* AWS Bedrock recipe notebook on standard, semantic, and hierarchical chunking [^20]


### 3.2  Advanced Strategies

* Late chunking and context-aware embeddings—Nvidia blog \& LinkedIn deep dive [^21][^22]
* Hierarchical chunking overview—BitPeak [^23]
* Reddit thread: turning PDFs into hierarchical structures for RAG [^24]


### 3.3  Libraries \& Code

* LangChain RecursiveCharacterTextSplitter \& SemanticChunker docs [^7][^25]
* IBM Granite tutorial on chunking with LangChain \& Chroma [^26]
* DataStax guide: JavaScript library `llm-chunk` example [^27]


## 4.  Vector Databases, Indexing \& Similarity Search

### 4.1  Conceptual Primers

* Weaviate blog: Vector search explained [^28]
* IBM “What is vector search?” overview [^29]
* Google BigQuery: Introduction to vector search [^30]


### 4.2  Database Options \& Tutorials

| Vector Store | Key Tutorial / Doc | Notes |
| :-- | :-- | :-- |
| **Faiss** | Pinecone “Faiss tutorial” [^31]; ProjectPro guide [^32]; Faiss docs [^33] | Local or GPU, billions of vectors |
| **Pinecone** | LangChain integration notebook [^34]; DataCamp Pinecone tutorial [^35]; AI-for-Devs quick-start [^36] | Fully-managed, hybrid search |
| **Weaviate** | Weaviate eBook “Advanced RAG Techniques” [^37] | Built-in modular chunkers |
| **Milvus** | Milvus DPR explainer [^38] | BM25 + vector hybrid examples |

### 4.3  Index Types \& Tuning

* HNSW, IVF, PQ—Faiss documentation [^33]
* Pinecone blog: composite indexes \& serverless scaling [^16]
* Vectorize docs: RAG pipeline connectors \& vectorization [^39]


## 5.  Retrieval Algorithms \& Hybrid Search

### 5.1  Dense Retrieval

* Dense Passage Retrieval original paper; GeeksforGeeks primer [^40]
* Paperspace blog: DPR with SimpleTransformers [^41]
* YouTube talk: Training DPR end-to-end [^42]


### 5.2  Sparse \& BM25

* Wikipedia Okapi BM25 formula [^43]
* LangChain BM25Retriever docs [^44]
* Milvus blog: BM25 in vector DBs [^45]


### 5.3  Hybrid \& Fusion

* DEV Community deep dive on hybrid retrieval (BM25 + FAISS) [^46]
* Microsoft Tech Community: vector + keyword fusion [^47]
* Superlinked article on hybrid search with reranking [^37]


## 6.  Putting It All Together

![Architecture diagram of Retrieval-Augmented Generation (RAG) showing the flow from user query through retrieval and extraction pipelines using NeMo components and vector databases.](https://pplx-res.cloudinary.com/image/upload/v1748546718/pplx_project_search_images/f4dba7ef927250f2463a6dfa3249aebeec1d6eeb.jpg)

Architecture diagram of Retrieval-Augmented Generation (RAG) showing the flow from user query through retrieval and extraction pipelines using NeMo components and vector databases.

### Essential Practice Tasks

1. **Build a toy RAG** with LangChain: load small docs, apply RecursiveCharacterTextSplitter, embed with `text-embedding-3-small`, index in FAISS, and query.
2. **Swap embeddings**: test E5-base vs OpenAI ADA 002, measure MRR with 20 synthetic questions.
3. **Try hybrid search**: combine BM25Retriever with vector retriever and compare answer faithfulness.
4. **Experiment with chunk sizes** (200 vs 500 tokens) and observe context-window utilization.

### Quick-Reference Cheat-Sheets

* Codecademy RAG foundations cheatsheet
* Nexla vector embedding overview [^48]
* Databricks glossary entry on RAG pipelines [^49]


## Conclusion

Mastering these Week 1 fundamentals—architecture flow, embedding choices, chunking heuristics, vector indexing, and retrieval algorithms—lays the groundwork for all later enhancements such as late chunking, reranking, or corrective RAG. Work through the readings, replicate the notebooks, and you will be equipped to build reliable, explainable retrieval-augmented systems ready for production.

<div style="text-align: center">⁂</div>

[^1]: https://arxiv.org/abs/2005.11401

[^2]: https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf

[^3]: https://arxiv.org/pdf/2312.10997.pdf

[^4]: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide

[^5]: https://www.ibm.com/architectures/patterns/genai-rag

[^6]: https://aws.amazon.com/what-is/retrieval-augmented-generation/

[^7]: https://python.langchain.com/docs/tutorials/rag/

[^8]: https://www.pinecone.io/learn/retrieval-augmented-generation/

[^9]: https://www.youtube.com/watch?v=yfHHvmaMkcA

[^10]: https://www.mongodb.com/developer/products/atlas/choose-embedding-model-rag/

[^11]: https://www.pinecone.io/learn/series/rag/embedding-models-rundown/

[^12]: https://www.datastax.com/blog/best-embedding-models-information-retrieval-2025

[^13]: https://huggingface.co/spaces/mteb/leaderboard

[^14]: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-generate-embeddings

[^15]: https://help.openai.com/en/articles/8868588-retrieval-augmented-generation-rag-and-semantic-search-for-gpts

[^16]: https://www.pinecone.io/learn/series/faiss/

[^17]: https://huggingface.co/docs/transformers/en/model_doc/dpr

[^18]: https://github.com/Hannibal046/nanoDPR

[^19]: https://unstructured.io/blog/chunking-for-rag-best-practices

[^20]: https://aws-samples.github.io/amazon-bedrock-samples/rag/open-source/chunking/rag_chunking_strategies_langchain_bedrock/

[^21]: https://www.linkedin.com/pulse/optimizing-rag-advanced-chunking-strategies-improved-poornachandra-yvs6f

[^22]: https://x.com/jerryjliu0/status/1761946367856152952?lang=en

[^23]: https://bitpeak.com/chunking-methods-in-rag-overview-of-available-solutions/

[^24]: https://www.reddit.com/r/LangChain/comments/1dpbc4g/how_we_chunk_turning_pdfs_into_hierarchical/

[^25]: https://www.sagacify.com/news/a-guide-to-chunking-strategies-for-retrieval-augmented-generation-rag

[^26]: https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai

[^27]: https://www.datastax.com/blog/how-to-chunk-text-in-javascript-for-rag-applications

[^28]: https://weaviate.io/blog/vector-search-explained

[^29]: https://www.ibm.com/think/topics/vector-search

[^30]: https://cloud.google.com/bigquery/docs/vector-search-intro

[^31]: https://www.pinecone.io/learn/series/faiss/faiss-tutorial/

[^32]: https://www.projectpro.io/article/faiss-vector-database/1009

[^33]: https://faiss.ai

[^34]: https://python.langchain.com/docs/integrations/vectorstores/pinecone/

[^35]: https://www.datacamp.com/tutorial/mastering-vector-databases-with-pinecone-tutorial

[^36]: https://www.ai-for-devs.com/blog/how

[^37]: https://weaviate.io/blog/vector-embeddings-explained

[^38]: https://milvus.io/ai-quick-reference/what-is-dense-passage-retrieval-and-how-does-it-improve-search

[^39]: https://docs.vectorize.io/concepts/rag-pipelines/

[^40]: https://www.geeksforgeeks.org/what-is-dense-passage-retrieval-dpr/

[^41]: https://blog.paperspace.com/dense-passage-retrieval/

[^42]: https://www.youtube.com/watch?v=QbeXGnuqgvY

[^43]: https://en.wikipedia.org/wiki/Okapi_BM25

[^44]: https://python.langchain.com/docs/integrations/retrievers/bm25/

[^45]: https://zilliz.com/learn/mastering-bm25-a-deep-dive-into-the-algorithm-and-application-in-milvus

[^46]: https://python.langchain.com/docs/integrations/vectorstores/faiss/

[^47]: https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview

[^48]: 1000074411.jpg

[^49]: https://nexla.com/ai-infrastructure/vector-embedding/

[^50]: https://www.signitysolutions.com/blog/rag-pipeline

[^51]: https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/

[^52]: https://www.multimodal.dev/post/rag-pipeline-diagram

[^53]: https://paperswithcode.com/method/rag

[^54]: https://humanloop.com/blog/rag-architectures

[^55]: https://customgpt.ai/components-of-a-rag-system/

[^56]: https://dl.acm.org/doi/abs/10.5555/3495724.3496517

[^57]: https://www.databricks.com/glossary/retrieval-augmented-generation-rag

[^58]: https://lakefs.io/blog/what-is-rag-pipeline/

[^59]: https://www.sciencedirect.com/science/article/pii/S2666920X25000578

[^60]: https://www.k2view.com/what-is-retrieval-augmented-generation

[^61]: https://www.datacamp.com/tutorial/introduction-to-vector-databases-for-machine-learning

[^62]: https://graphrag.com/guides/chunking/

[^63]: https://myscale.com/blog/easy-vector-retrieval-faiss-python/

[^64]: https://www.youtube.com/watch?v=0jOlZpFFxCE

[^65]: https://nexla.com/ai-infrastructure/vector-databases/

[^66]: https://addaxis.ai/advanced-chunking-strategies-for-rag/

[^67]: https://towardsdatascience.com/building-an-image-similarity-search-engine-with-faiss-and-clip-2211126d08fa/

[^68]: https://www.pinecone.io/learn/vector-database/

[^69]: https://www.f22labs.com/blogs/7-chunking-strategies-in-rag-you-need-to-know/

[^70]: https://unfoldai.com/effortless-large-scale-image-retrieval-with-faiss-a-hands-on-tutorial/

[^71]: https://realpython.com/chromadb-vector-database/

[^72]: https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/

[^73]: https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.vectorstorepinecone/

[^74]: https://www.e2enetworks.com/blog/dense-passage-retrieval-for-open-domain-question-answering

[^75]: https://www.youtube.com/watch?v=erUfLIi9OFM

[^76]: https://www.geeksforgeeks.org/what-is-bm25-best-matching-25-algorithm/

[^77]: https://www.luigisbox.com/search-glossary/bm25/

[^78]: https://www.youtube.com/watch?v=v4bye5Rfa3g

[^79]: https://web.stanford.edu/class/cs276/handouts/lecture12-bm25etc.pdf

[^80]: https://www.pingcap.com/article/mastering-faiss-vector-database-a-beginners-handbook/

[^81]: https://arxiv.org/html/2407.08275v1

[^82]: https://www.singlestore.com/blog/a-guide-to-retrieval-augmented-generation-rag/

[^83]: https://www.reddit.com/r/LangChain/comments/1blfg7i/what_is_the_current_best_embedding_model_for/

[^84]: https://nexla.com/ai-infrastructure/retrieval-augmented-generation/

[^85]: https://softwaremill.com/embedding-models-comparison/

[^86]: https://github.com/facebookresearch/faiss

[^87]: https://www.youtube.com/watch?v=3zYtfqxi6EU

[^88]: https://www.elastic.co/what-is/vector-search

[^89]: https://learn.microsoft.com/en-us/azure/search/vector-search-overview

[^90]: https://www.datacamp.com/blog/faiss-facebook-ai-similarity-search

[^91]: https://www.pinecone.io/learn/vector-search-basics/

[^92]: https://www.coveo.com/blog/what-is-vector-search/

