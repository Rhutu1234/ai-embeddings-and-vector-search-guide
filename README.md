# Embeddings & Vector Search

*A deep-dive walkthrough of embeddings and vector search — covering what embeddings are and how text embeddings are produced, semantic similarity and cosine similarity (with the math and the .NET code), exact vs. approximate similarity search, what vector databases are for, hands-on treatment of Azure AI Search, PostgreSQL with pgvector, Redis vector search, and MongoDB vector search, and hybrid search — combining keyword and vector retrieval with Reciprocal Rank Fusion.*

---

## Table of Contents

1. [Introduction](#introduction)
2. [What Are Embeddings?](#1-what-are-embeddings)
3. [Text Embeddings](#2-text-embeddings)
4. [Semantic Similarity](#3-semantic-similarity)
5. [Cosine Similarity](#4-cosine-similarity)
6. [Similarity Search](#5-similarity-search)
7. [Vector Databases](#6-vector-databases)
8. [Azure AI Search](#7-azure-ai-search)
9. [PostgreSQL + pgvector](#8-postgresql--pgvector)
10. [Redis Vector Search](#9-redis-vector-search)
11. [MongoDB Vector Search](#10-mongodb-vector-search)
12. [Hybrid Search](#11-hybrid-search)
13. [Choosing a Vector Store](#12-choosing-a-vector-store)
14. [Putting It Together: An Ingestion and Query Pipeline](#13-putting-it-together-an-ingestion-and-query-pipeline)
15. [Common Pitfalls](#14-common-pitfalls)
16. [Quick Reference Table](#quick-reference-table)
17. [Conclusion](#conclusion)

---

## Introduction

Traditional search matches **words**: a query for "reset my password" finds documents containing those words, and misses a perfectly relevant page titled "Recovering account access." Vector search matches **meaning**. It does this by converting text (or images, audio, products, users) into lists of numbers called **embeddings**, arranged so that items with similar meaning end up close together. Finding relevant content then becomes a geometry problem: find the nearest points to the query.

```csharp
EmbeddingClient client = new("text-embedding-3-small", apiKey);

ReadOnlyMemory<float> query = client.GenerateEmbedding("How do I reset my password?").Value.ToFloats();
ReadOnlyMemory<float> doc1  = client.GenerateEmbedding("Steps for recovering account access").Value.ToFloats();
ReadOnlyMemory<float> doc2  = client.GenerateEmbedding("Quarterly financial results for Q3").Value.ToFloats();

float s1 = TensorPrimitives.CosineSimilarity(query.Span, doc1.Span);   // high — same meaning, different words
float s2 = TensorPrimitives.CosineSimilarity(query.Span, doc2.Span);   // low  — unrelated
```

That is the whole idea: embed, compare, rank. Everything else in this guide is about doing it **at scale** (millions of vectors), **fast** (approximate indexes), **correctly** (filtering, hybrid search), and **inside the database you already run** — or a dedicated one. Embeddings are also the foundation of retrieval-augmented generation (RAG), which this series' *LLM Application Development* guide covers as an application pattern; this guide is the retrieval engine underneath it. The concept of embeddings is introduced in this series' *AI/ML Fundamentals* guide.

> **A note on versions and model names.** Embedding model names, dimensions, SDK method names, and database feature availability change often. Model names here (`text-embedding-3-small` and so on) are examples, and the SDK samples target recent package versions — verify against the versions you install and the current documentation of each product.

---

## 1. What Are Embeddings?

### A learned numeric fingerprint of meaning

An **embedding** is a fixed-length list of floating-point numbers (a *vector*) that represents an item so that **similar items have similar vectors**.

```plaintext
"dog"      ->  [ 0.12, -0.48,  0.33, ...,  0.07 ]     (hundreds to thousands of numbers)
"puppy"    ->  [ 0.10, -0.45,  0.35, ...,  0.05 ]     <- very close to "dog"
"invoice"  ->  [-0.61,  0.22, -0.09, ..., 0.40 ]      <- far from both
```

You can think of each embedding as a **point in a high-dimensional space** (say, 1,536 dimensions). Nobody assigns the individual numbers meaning — a neural network *learns* them during training so that the geometry encodes semantic relationships: items used in similar contexts land near each other.

### Why not just use the words themselves?

```plaintext
Keyword / one-hot representation:
  "cat" and "kitten" share NO columns — to the machine they are exactly as unrelated as "cat" and "tractor".

Embedding representation:
  "cat" and "kitten" are NEARBY points; "cat" and "tractor" are distant.
  Meaning becomes a measurable distance.
```

### What can be embedded

```plaintext
Text         words, sentences, paragraphs, documents, code
Images       photos, product pictures (image models; also multimodal models that embed
             images and text into the SAME space, enabling text-to-image search)
Audio        speech, music
Structured   products, users, items (recommendation embeddings — see this series' ML.NET guide)
```

This guide focuses on **text embeddings**, the most common case, but every technique here (similarity metrics, indexes, vector databases) works identically for any embedding.

### The dimensionality of an embedding

```plaintext
Dimensions = how many numbers are in the vector (a property of the MODEL, e.g. 384, 768, 1536, 3072).
More dimensions can capture finer distinctions, but cost more storage, memory, and search time.
Every vector in one index must have the SAME number of dimensions, from the SAME model.
```

---

## 2. Text Embeddings

### Producing embeddings: a model call

A **text embedding model** takes a string and returns its vector. It is a different kind of model from a chat model: it doesn't generate text, it just encodes it — and it's typically far cheaper and faster per token.

```csharp
using OpenAI.Embeddings;

EmbeddingClient client = new("text-embedding-3-small", apiKey);

// One text
OpenAIEmbedding embedding = client.GenerateEmbedding("How do I reset my password?");
ReadOnlyMemory<float> vector = embedding.ToFloats();       // e.g. 1,536 floats

// Many texts in one request (far more efficient for ingestion)
var texts = new List<string> { "First passage...", "Second passage...", "Third passage..." };
OpenAIEmbeddingCollection results = (await client.GenerateEmbeddingsAsync(texts)).Value;
foreach (OpenAIEmbedding e in results)
    Store(e.Index, e.ToFloats());
```

On Azure OpenAI, only the client construction changes (use the **deployment name** of your embedding model):

```csharp
EmbeddingClient azureEmbeddings = azureOpenAiClient.GetEmbeddingClient("my-embedding-deployment");
```

### Shortening dimensions

```csharp
var options = new EmbeddingGenerationOptions { Dimensions = 512 };
OpenAIEmbedding smaller = client.GenerateEmbedding(text, options);
```

Some modern embedding models can return shortened vectors with only modest quality loss — a useful lever for reducing storage and speeding up search, and sometimes necessary to fit a database's index limits (Section 8). Check whether your model supports it.

### Where embedding models come from

```plaintext
Hosted APIs        OpenAI, Azure OpenAI, Cohere, Google, and others — simplest to operate,
                   billed per token.
Open-source        Sentence-transformer style models you can run yourself (including via ONNX
                   in .NET) — control and privacy, no per-token fee, but you operate the
                   infrastructure.
```

### Rules of embedding hygiene

```plaintext
1. USE THE SAME MODEL for documents and queries. Vectors from different models live in
   different spaces and are NOT comparable — not even if the dimensions happen to match.

2. RECORD THE MODEL NAME/VERSION alongside stored vectors. Switching models later means
   RE-EMBEDDING EVERYTHING. Store the model identifier so you can tell which vectors are stale.

3. CHUNK LONG DOCUMENTS. Models have input-length limits, and one vector for a 50-page
   document blurs everything into mush. Split into passages (commonly a few hundred tokens),
   often with slight overlap so ideas aren't cut in half, and embed each chunk separately.
   Keep a reference from each chunk back to its source and position.

4. ADD CONTEXT TO CHUNKS where useful (document title, section heading) so a chunk that says
   "it must be renewed annually" still carries what "it" refers to.

5. BATCH your embedding calls during ingestion, and respect rate limits (see this series'
   AI APIs with .NET guide).

6. NORMALIZATION: many embedding models return vectors already scaled to length 1 (unit
   vectors). For unit vectors, cosine similarity and dot product give identical rankings
   (Section 4). Check your model's documentation.
```

### Embeddings are data — treat them as sensitive

```plaintext
An embedding is derived from your text, and research has shown that some original content can be
partly reconstructed from embeddings. Apply the same access control, encryption, and retention
policy to vectors as to the source documents. Also remember that sending text to a hosted
embedding API sends that text to the provider.
```

---

## 3. Semantic Similarity

### Closeness in the vector space ≈ closeness in meaning

**Semantic similarity** is the idea that two pieces of text are similar if they *mean* similar things, regardless of the exact words used.

```plaintext
Query:      "How can I get my money back?"

Keyword search finds:        documents containing "money" and "back"
Semantic search also finds:  "Refund policy"            (no shared words)
                             "Returning a purchase"      (no shared words)
                             "Cancel and reimbursement"  (no shared words)
```

This is why vector search transformed search, recommendations, deduplication, and RAG: users phrase things in endless ways, and meaning-based matching handles synonyms, paraphrase, and even (with the right model) other languages.

### Where semantic similarity shines

```plaintext
- Question answering over documentation (the retrieval step of RAG)
- Searching by concept: "tools for tracking my spending"
- Recommending similar articles, products, or support tickets
- Detecting near-duplicates and clustering related content
- Cross-lingual search with multilingual embedding models
- Matching a user's request to the right tool, FAQ, or intent
```

### Where it is weak — know the failure modes

```plaintext
EXACT TOKENS     Product codes, error numbers, names, and IDs ("ERR-40412", "SKU 88-ZK")
                 are weak signals to an embedding model; keyword search nails them.
                 (This is the main reason for hybrid search, Section 11.)

NEGATION AND     "Cancel my subscription" and "Don't cancel my subscription" can embed
DETAIL           very close together; embeddings capture topic more reliably than
                 precise logical meaning.

NUMBERS AND      "Revenue of $5M" vs. "Revenue of $50M" may look nearly identical.
DATES

DOMAIN SHIFT     A general model may represent specialized jargon (legal, medical,
                 internal acronyms) poorly.

"SIMILAR" ISN'T  The nearest chunk may be topically related but not actually contain the answer.
"RELEVANT"       Similarity is a retrieval heuristic, not a guarantee.
```

Evaluate retrieval on **your** data with real queries rather than assuming it works (Section 13).

---

## 4. Cosine Similarity

### The standard way to measure how alike two vectors are

**Cosine similarity** measures the **angle** between two vectors, ignoring their length. Vectors pointing in the same direction score 1; unrelated (perpendicular) vectors score about 0; opposite directions score -1.

```plaintext
                    A · B                    sum of (A[i] * B[i])
cos(A, B)  =  ───────────────  =  ──────────────────────────────────────
               ||A|| × ||B||      sqrt(sum A[i]²)  ×  sqrt(sum B[i]²)
```

### A worked example in two dimensions

```plaintext
A = [1, 0]      B = [1, 1]

A · B  = (1×1) + (0×1)              = 1
||A||  = sqrt(1² + 0²)              = 1
||B||  = sqrt(1² + 1²)              = 1.414
cos    = 1 / (1 × 1.414)            = 0.707          (a 45° angle)

A = [1, 2, 3]   B = [2, 4, 6]       (same direction, B is just twice as long)
cos    = 1.0                                          (length is ignored — only direction matters)
```

### Why angle rather than distance?

```plaintext
For text embeddings, DIRECTION carries the meaning; length is mostly an artifact.
Cosine similarity compares meaning without being thrown off by magnitude — for example,
a long document and a short summary of it can point the same way even though their
raw vectors differ in size.
```

### In .NET

```csharp
using System.Numerics.Tensors;

float similarity = TensorPrimitives.CosineSimilarity(a.Span, b.Span);   // hardware-accelerated
```

Or by hand, which is worth seeing once:

```csharp
static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
{
    float dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.Length; i++)
    {
        dot   += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB));
}
```

### The three common metrics — and the naming trap

```plaintext
COSINE SIMILARITY    angle between vectors. Range -1..1, HIGHER = more similar.
DOT PRODUCT          sum of A[i]*B[i]. For UNIT-LENGTH vectors it equals cosine similarity
                     and ranks identically — and it's slightly cheaper to compute.
EUCLIDEAN (L2)       straight-line distance between points. LOWER = more similar.
                     For unit vectors, L2 distance and cosine similarity also give the same ranking.

THE NAMING TRAP: databases often expose DISTANCES (smaller is better), not similarities.
  cosine DISTANCE   = 1 - cosine similarity     (0 = identical, larger = less similar)
  pgvector operators: <=> cosine distance,  <-> L2 distance,  <#> NEGATIVE inner product
  Mixing up "higher is better" with "lower is better" produces results that are exactly backwards.
```

**Choose the metric your embedding model was trained for** — for most text embedding models that's cosine similarity (or the equivalent dot product on normalized vectors). Create the index with the same metric you query with.

---

## 5. Similarity Search

### The task: given a query vector, find the k most similar stored vectors

This is **k-nearest-neighbor (kNN) search**, usually phrased "top-k."

### Exact search: compare against everything

```csharp
// Brute-force kNN: score every stored vector, keep the best k
var topK = documents
    .Select(d => (d.Id, Score: TensorPrimitives.CosineSimilarity(query.Span, d.Vector.Span)))
    .OrderByDescending(x => x.Score)
    .Take(5)
    .ToList();
```

```plaintext
EXACT (brute-force) search
  + Perfect recall — always the true nearest neighbors
  + No index to build or tune
  - Cost grows linearly with the number of vectors: every query touches every vector
  OK for thousands to low hundreds of thousands of vectors, depending on dimensions and hardware.
```

### Approximate search (ANN): trade a little accuracy for a lot of speed

At millions of vectors, scanning everything per query is too slow. **Approximate nearest neighbor (ANN)** indexes organize vectors so a query examines only a small, promising fraction — returning *almost always* the true nearest neighbors, in a fraction of the time.

```plaintext
HNSW (Hierarchical Navigable Small World)
  A layered graph of vectors; search hops from a coarse layer down to fine ones, greedily
  moving toward the query. Excellent speed/recall; the default in most systems.
  Costs: memory-hungry, slower to build, and updates are more expensive than a simple scan.
  Knobs: m (connections per node), ef_construction (build quality),
         ef_search (how wide the query explores — higher = better recall, slower).

IVF (Inverted File)
  Cluster the vectors; at query time search only the few clusters nearest the query.
  Cheaper to build and smaller than HNSW; recall depends on how many clusters you probe.
  Knobs: number of lists/clusters, number to probe.

Quantization / compression
  Store vectors with fewer bits (e.g. 8-bit or binary) to cut memory dramatically,
  at some cost to precision; often combined with re-scoring the top candidates on the
  full-precision vectors.
```

### The tuning triangle

```plaintext
        RECALL  (did we find the true nearest neighbors?)
         /  \
        /    \
    SPEED ── MEMORY/COST

You can't maximize all three. Higher recall means exploring more (slower) or storing
more (bigger). Tune the knobs against YOUR data using a recall measurement — compare
ANN results to exact results on a sample of queries.
```

### Search is more than "top-k"

```plaintext
FILTERING        "Nearest vectors WHERE tenant = 'acme' AND category = 'docs'."
                 Essential for multi-tenant apps and access control. Filtering and ANN interact
                 subtly: apply the filter BEFORE or DURING the vector search (pre-filtering) rather
                 than after, or you may get fewer than k results (Section 14).

SCORE THRESHOLD  Drop results below a minimum similarity so the system can say "no good match"
                 instead of returning the least-bad one. Tune the threshold on real queries —
                 scales differ between models and metrics.

TOP-K SIZING     Retrieve more candidates than you'll finally use (say 20-50), then rerank or
                 filter down to the best few.

RERANKING        A second-stage model (a cross-encoder or a semantic ranker) rescores the top
                 candidates by reading the query and each passage TOGETHER — slower but more
                 accurate than vector distance alone.
```

---

## 6. Vector Databases

### What a vector database does

A **vector database** (or any database with vector capabilities) stores embeddings alongside their metadata and answers "find the nearest vectors" queries efficiently using ANN indexes, with filtering, updates, and persistence.

```plaintext
Core capabilities:
  - Store vectors + payload (the text chunk, source, tenant, tags, timestamps)
  - ANN index (HNSW / IVF / ...) for fast nearest-neighbor queries
  - Metadata filtering combined with vector search
  - Insert / update / delete without rebuilding everything
  - Scale, replication, backup, and access control (varies widely by product)
  - Often: hybrid search, reranking, and integrated embedding generation
```

### Two families

```plaintext
PURPOSE-BUILT VECTOR DATABASES   (e.g. Pinecone, Qdrant, Weaviate, Milvus, Chroma)
  Designed around vectors from the start; rich vector-specific features and tuning.
  Cost: another system to deploy, secure, operate, back up, and keep in sync with your primary data.

VECTOR CAPABILITIES IN A DATABASE YOU ALREADY RUN
  Azure AI Search, PostgreSQL (pgvector), Redis, MongoDB Atlas, SQL Server / Azure SQL,
  Azure Cosmos DB, Elasticsearch/OpenSearch, and others.
  Vectors live next to your existing data, share its transactions/security/backups,
  and add no new infrastructure — and usually scale well past what most applications need.
```

### Which to pick (the honest guidance)

```plaintext
Start with what you already operate. If your data is in PostgreSQL, try pgvector first.
If you already run Azure AI Search for text search, add vectors there. A dedicated vector
database earns its place when you have very large scale (hundreds of millions of vectors or
more), demanding latency/throughput targets, or need specialized vector features your
existing database lacks. Don't add infrastructure to solve a problem you don't have yet.
```

The next four sections cover the options most relevant to a .NET team — each with the same sequence: **define the index → store vectors → query**.

---

## 7. Azure AI Search

### A managed search service with first-class vector and hybrid search

Azure AI Search (formerly Azure Cognitive Search) is Microsoft's managed search service. It combines full-text search (BM25), vector search, filtering, and optional semantic reranking in one service — which makes it a strong fit for **hybrid** retrieval (Section 11) and for .NET teams already on Azure.

### Define an index with a vector field

```csharp
using Azure;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.Search.Documents.Models;

var indexClient = new SearchIndexClient(new Uri(endpoint), new AzureKeyCredential(adminKey));

var index = new SearchIndex("docs-index")
{
    Fields =
    {
        new SimpleField("id", SearchFieldDataType.String) { IsKey = true },
        new SearchableField("content") { AnalyzerName = LexicalAnalyzerName.EnMicrosoft },
        new SimpleField("category", SearchFieldDataType.String) { IsFilterable = true },
        new VectorSearchField("contentVector", vectorSearchDimensions: 1536, vectorSearchProfileName: "vector-profile"),
    },
    VectorSearch = new VectorSearch
    {
        Profiles  = { new VectorSearchProfile("vector-profile", "hnsw-config") },
        Algorithms =
        {
            new HnswAlgorithmConfiguration("hnsw-config")
            {
                Parameters = new HnswParameters { Metric = VectorSearchAlgorithmMetric.Cosine }
            }
        }
    }
};

await indexClient.CreateOrUpdateIndexAsync(index);
```

The dimension count (1536) must match your embedding model's output exactly. The **profile** ties the field to an algorithm configuration (HNSW here, with the cosine metric).

### Upload documents with their vectors

```csharp
public class DocItem
{
    public string Id { get; set; } = "";
    public string Content { get; set; } = "";
    public string Category { get; set; } = "";
    public ReadOnlyMemory<float> ContentVector { get; set; }
}

var searchClient = new SearchClient(new Uri(endpoint), "docs-index", new AzureKeyCredential(queryOrAdminKey));

var batch = chunks.Select((chunk, i) => new DocItem
{
    Id = i.ToString(),
    Content = chunk.Text,
    Category = chunk.Category,
    ContentVector = embeddings[i],
}).ToList();

await searchClient.UploadDocumentsAsync(batch);
```

### Query: pure vector search

```csharp
var vectorQuery = new VectorizedQuery(queryEmbedding)       // ReadOnlyMemory<float> from your embedding call
{
    KNearestNeighborsCount = 5,
    Fields = { "contentVector" }
};

var options = new SearchOptions
{
    VectorSearch = new() { Queries = { vectorQuery } },
    Filter = "category eq 'docs'",                           // OData filter, combined with vector search
    Size = 5,
    Select = { "id", "content", "category" }
};

SearchResults<DocItem> results = await searchClient.SearchAsync<DocItem>(null, options);   // null text = vector only

await foreach (SearchResult<DocItem> r in results.GetResultsAsync())
    Console.WriteLine($"{r.Score:F3}  {r.Document.Content}");
```

### Query: hybrid search — add the text query

```csharp
SearchResults<DocItem> hybrid = await searchClient.SearchAsync<DocItem>(
    "how do I reset my password",                            // keyword (BM25) leg
    options);                                                // vector leg (the same options)
```

Passing both a text query and a vector query makes Azure AI Search run the two retrievals in parallel and merge them with **Reciprocal Rank Fusion** (Section 11) into one ranked list.

### Useful features beyond the basics

```plaintext
SEMANTIC RANKER      An optional reranking stage that rescored the top results with a language
                     model for better relevance. Hybrid retrieval + semantic ranking is
                     Microsoft's commonly recommended configuration for RAG.
INTEGRATED           Indexers and skillsets can chunk documents and generate embeddings at
VECTORIZATION        indexing time, and vectorize query text at query time, so you don't hand-roll
                     the pipeline.
FILTERS              OData filter expressions work alongside vector queries; for vector queries
                     the default is to filter BEFORE the vector search.
SCALE UNITS          Replicas and partitions scale throughput and storage; vector capacity and
                     limits depend on the service tier — check current limits before sizing.
```

### Trade-offs

```plaintext
+ Managed (no servers to run), hybrid + semantic ranking built in, rich filtering,
  good .NET SDK, strong Azure security integration (Entra ID, private endpoints).
- Billed by service tier/units rather than per query, so cost is a baseline even when idle;
  data lives in a separate search index that you must keep in sync with your system of record.
```

---

## 8. PostgreSQL + pgvector

### Vector search inside the database you probably already have

**pgvector** is an open-source PostgreSQL extension adding a `vector` data type, distance operators, and ANN indexes. Because vectors live in ordinary tables, you get **SQL joins, transactions, filters, and your existing backup and security model**. It's available on most managed PostgreSQL services (check your provider's supported version).

### Schema and data

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id         BIGSERIAL PRIMARY KEY,
    content    TEXT        NOT NULL,
    category   TEXT        NOT NULL,
    tenant_id  INT         NOT NULL,
    embedding  VECTOR(1536) NOT NULL         -- dimension must match your embedding model
);

INSERT INTO documents (content, category, tenant_id, embedding)
VALUES ('Steps for recovering account access', 'docs', 1, '[0.012, -0.034, 0.101, ...]');
```

### Query: nearest neighbors with an operator

```sql
-- $1 is the query embedding. <=> is COSINE DISTANCE (smaller = more similar).
SELECT id, content, 1 - (embedding <=> $1) AS similarity
FROM documents
WHERE tenant_id = 1 AND category = 'docs'
ORDER BY embedding <=> $1
LIMIT 5;
```

```plaintext
Operators:   <=>  cosine distance      <->  L2 (Euclidean) distance      <#>  NEGATIVE inner product
Convert distance to similarity for display:  similarity = 1 - cosine_distance
The query must ORDER BY the distance operator with a LIMIT to be able to use a vector index.
```

### Indexing: from exact scan to ANN

```sql
-- With no index, every query is an EXACT scan: perfect recall, slow at scale.

-- HNSW: best query speed/recall; slower to build, more memory
CREATE INDEX documents_embedding_hnsw
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- IVFFlat: faster to build, smaller; build AFTER loading representative data
CREATE INDEX documents_embedding_ivf
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Query-time recall/speed trade-off
SET hnsw.ef_search = 100;       -- HNSW: wider exploration, better recall, slower
SET ivfflat.probes = 10;        -- IVFFlat: probe more clusters, better recall, slower
```

```plaintext
The operator class in the index (vector_cosine_ops) must MATCH the operator in your query (<=>).
  vector_cosine_ops -> <=>     vector_l2_ops -> <->     vector_ip_ops -> <#>
A mismatch silently falls back to a full scan.

DIMENSION LIMIT: the plain vector type can be indexed only up to a certain number of dimensions
(2,000 in recent pgvector versions). A 3,072-dimension embedding can't be indexed as-is — shorten
the embedding (Section 2), or use a smaller-precision type such as halfvec, whose indexable
limit is higher. Check the current pgvector documentation.
```

### Filtered search: the pgvector gotcha

```plaintext
With an ANN index, the index returns its nearest candidates FIRST and the WHERE filter is
applied AFTERWARD. If the filter is selective (one small tenant among many), most candidates are
filtered out and you may get FEWER than k rows — or none — even though matching rows exist.

Remedies:
  - Raise hnsw.ef_search so the index returns more candidates before filtering.
  - Newer pgvector versions offer iterative index scans that keep scanning until enough rows
    pass the filter — check the version you run.
  - Use PARTIAL indexes (CREATE INDEX ... WHERE tenant_id = 1) or PARTITION by tenant for
    highly selective, high-value filters.
  - Ordinary B-tree indexes on the filter columns still help when the filter is very selective.
```

### From .NET: Npgsql + the Pgvector package

```csharp
// dotnet add package Npgsql
// dotnet add package Pgvector
using Npgsql;
using Pgvector;

var dataSourceBuilder = new NpgsqlDataSourceBuilder(connectionString);
dataSourceBuilder.UseVector();                                   // registers the vector type
await using var dataSource = dataSourceBuilder.Build();
await using var conn = await dataSource.OpenConnectionAsync();

float[] queryVector = queryEmbedding.ToArray();

await using var cmd = new NpgsqlCommand(
    "SELECT id, content, 1 - (embedding <=> $1) AS similarity " +
    "FROM documents WHERE tenant_id = $2 ORDER BY embedding <=> $1 LIMIT 5", conn);
cmd.Parameters.AddWithValue(new Vector(queryVector));
cmd.Parameters.AddWithValue(tenantId);

await using var reader = await cmd.ExecuteReaderAsync();
while (await reader.ReadAsync())
    Console.WriteLine($"{reader.GetDouble(2):F3}  {reader.GetString(1)}");
```

Always **parameterize** — never build SQL by concatenating text (see this series' SQL guides). The query vector is passed as a typed parameter.

### From EF Core

```csharp
// dotnet add package Pgvector.EntityFrameworkCore
using Pgvector;
using Pgvector.EntityFrameworkCore;

public class Document
{
    public long Id { get; set; }
    public string Content { get; set; } = "";
    public int TenantId { get; set; }

    [Column(TypeName = "vector(1536)")]
    public Vector Embedding { get; set; } = null!;
}

// DbContext configuration
optionsBuilder.UseNpgsql(connectionString, o => o.UseVector());
// modelBuilder.HasPostgresExtension("vector");

var nearest = await context.Documents
    .Where(d => d.TenantId == tenantId)
    .OrderBy(d => d.Embedding.CosineDistance(new Vector(queryVector)))
    .Take(5)
    .ToListAsync();
```

This is the same EF Core translation model this series' EF Core guide describes — the `CosineDistance` call becomes the `<=>` operator in SQL.

### Trade-offs

```plaintext
+ No new infrastructure; vectors next to relational data; full SQL (joins, filters, transactions);
  one backup, one security model; great for tens of thousands to a few million vectors, and often more.
- ANN-with-filter quirks (above); index builds and memory use grow with data; at very large scale
  or extreme throughput, a dedicated engine may do better; hybrid search is DIY (Section 11).
```

---

## 9. Redis Vector Search

### Very low-latency vector search, in memory

Redis's query engine (available in Redis Stack, Redis Enterprise/Cloud, and Azure's Redis offerings that include the search module — verify support for your tier) can index **vectors stored in hashes or JSON documents** and run kNN queries combined with filters. Because data lives in memory, latency is excellent — and so is the cost per gigabyte.

### Create an index

```plaintext
FT.CREATE idx:docs ON HASH PREFIX 1 doc:
  SCHEMA
    content   TEXT
    category  TAG
    embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE
```

Any hash whose key starts with `doc:` is indexed automatically. The vector field declares the algorithm (`HNSW`, or `FLAT` for exact search), the element type, the dimension count, and the metric.

### Store and query from .NET (StackExchange.Redis + NRedisStack)

```csharp
// dotnet add package NRedisStack
using NRedisStack;
using NRedisStack.RedisStackCommands;
using NRedisStack.Search;
using NRedisStack.Search.Literals.Enums;
using StackExchange.Redis;
using System.Runtime.InteropServices;

var redis = await ConnectionMultiplexer.ConnectAsync(connectionString);
var db = redis.GetDatabase();

// 1. Create the index
var schema = new Schema()
    .AddTextField("content")
    .AddTagField("category")
    .AddVectorField("embedding", Schema.VectorField.VectorAlgo.HNSW,
        new Dictionary<string, object>
        {
            ["TYPE"] = "FLOAT32",
            ["DIM"] = 1536,
            ["DISTANCE_METRIC"] = "COSINE"
        });

await db.FT().CreateAsync("idx:docs",
    new FTCreateParams().On(IndexDataType.HASH).Prefix("doc:"), schema);

// 2. Store a document — the vector is stored as RAW float32 BYTES
byte[] ToBytes(ReadOnlyMemory<float> v) => MemoryMarshal.AsBytes(v.Span).ToArray();

await db.HashSetAsync("doc:1", new HashEntry[]
{
    new("content", "Steps for recovering account access"),
    new("category", "docs"),
    new("embedding", ToBytes(docEmbedding)),
});

// 3. Query: pre-filter by tag, then 5 nearest neighbors
var query = new Query("(@category:{docs})=>[KNN 5 @embedding $vec AS score]")
    .AddParam("vec", ToBytes(queryEmbedding))
    .SetSortBy("score")
    .ReturnFields("content", "score")
    .Dialect(2);

SearchResult result = await db.FT().SearchAsync("idx:docs", query);
foreach (var doc in result.Documents)
    Console.WriteLine($"{doc["score"]}  {doc["content"]}");
```

```plaintext
Details that matter:
  - The returned "score" is a DISTANCE (for cosine: smaller = more similar), not a similarity.
  - Vectors must be passed as binary float32 blobs, in the same byte layout used at write time.
  - The filter in parentheses is applied as a PRE-filter to the KNN search.
  - Use DIALECT 2 (or later) for vector queries.
```

### Redis's sweet spots

```plaintext
SEMANTIC CACHING    Embed incoming prompts; if a near-identical question was answered recently,
                    return the cached LLM response instead of calling the model — cutting cost
                    and latency (see this series' AI APIs with .NET guide on cost optimization).
SESSION / CHAT      Fast retrieval of conversation memory and recent context
MEMORY              (see this series' LLM Application Development guide).
REAL-TIME           Recommendations and lookups where millisecond latency matters, on a
                    dataset that fits in memory.
```

### Trade-offs

```plaintext
+ Excellent latency; already present in many stacks; flexible data structures;
  hybrid-style queries mixing TEXT/TAG/NUMERIC filters with KNN.
- Memory is expensive: large vector sets cost much more than on disk-based systems
  (quantization helps); durability depends on persistence settings you must configure
  deliberately; not a general system of record for your documents.
```

---

## 10. MongoDB Vector Search

### Vector search alongside your documents (Atlas Vector Search)

MongoDB Atlas supports vector search through a dedicated **vector search index** and the **`$vectorSearch`** aggregation stage. Embeddings are stored as plain arrays inside your documents — right next to the rest of the record — which suits applications already modeled as documents. (Vector search availability for self-managed MongoDB deployments has been evolving; check current documentation for your deployment type.)

### Define the index

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine" },
    { "type": "filter", "path": "category" },
    { "type": "filter", "path": "tenantId" }
  ]
}
```

Create it in the Atlas UI, the CLI/API, or via the driver. Note that any field you want to **filter on** during vector search must be declared as a `filter` field in the index.

### Query with `$vectorSearch`

```javascript
db.documents.aggregate([
  {
    $vectorSearch: {
      index: "vector_index",
      path: "embedding",
      queryVector: [0.012, -0.034, 0.101 /* ... */],
      numCandidates: 150,        // how many candidates the ANN search considers
      limit: 5,                  // how many results to return
      filter: { category: "docs", tenantId: 1 }
    }
  },
  { $project: { content: 1, category: 1, score: { $meta: "vectorSearchScore" } } }
])
```

### From .NET (MongoDB.Driver)

```csharp
// dotnet add package MongoDB.Driver
using MongoDB.Bson;
using MongoDB.Driver;

var collection = database.GetCollection<BsonDocument>("documents");

var pipeline = new[]
{
    new BsonDocument("$vectorSearch", new BsonDocument
    {
        { "index", "vector_index" },
        { "path", "embedding" },
        { "queryVector", new BsonArray(queryEmbedding.ToArray().Select(f => (double)f)) },
        { "numCandidates", 150 },
        { "limit", 5 },
        { "filter", new BsonDocument { { "category", "docs" }, { "tenantId", 1 } } }
    }),
    new BsonDocument("$project", new BsonDocument
    {
        { "content", 1 },
        { "score", new BsonDocument("$meta", "vectorSearchScore") }
    })
};

List<BsonDocument> results = await collection.Aggregate<BsonDocument>(pipeline).ToListAsync();
```

Recent versions of the driver also offer a strongly typed `VectorSearch` helper on the aggregation builder — check the driver version you use. The raw pipeline above works regardless.

### Details that matter

```plaintext
numCandidates vs. limit
  numCandidates is how many nearest candidates the approximate search examines; limit is how many
  you get back. A larger numCandidates (commonly 10-20x the limit) improves recall at the cost of
  latency. Tune it like ef_search in HNSW.

SCORE
  vectorSearchScore is a SIMILARITY normalized so higher = better — unlike pgvector's and Redis's
  distances. Check each system's convention.

FILTERS
  Filter fields must be indexed as type "filter". Filters are applied during the search rather
  than after, so selective filters generally still return a full result set.

KEEPING VECTORS IN SYNC
  Update the embedding whenever the source text changes — a stale vector silently returns
  results for the OLD text. Atlas can also vectorize automatically in some configurations;
  check what your tier offers.
```

### Trade-offs

```plaintext
+ Vectors live inside the document with its data; flexible schema; scales horizontally;
  hybrid options via Atlas Search; natural fit if you already run MongoDB.
- Atlas-centric (check deployment support); index definitions and filter fields are managed
  separately from the collection; cost tied to dedicated search nodes at scale.
```

---

## 11. Hybrid Search

### Combine meaning-matching and word-matching

Vector search and keyword search have **opposite strengths** (Section 3):

```plaintext
KEYWORD (BM25 / full-text)               VECTOR (semantic)
  + exact terms, IDs, codes, names         + synonyms, paraphrase, intent
  + explainable                            + cross-lingual
  - misses different wording               - weak on rare tokens, exact strings, numbers
```

**Hybrid search** runs both retrievals and **fuses** the results — typically outperforming either alone, especially on real-world queries that mix concepts and specifics ("error 40412 when resetting password").

```plaintext
          ┌──> keyword search (BM25)  ──> ranked list A ──┐
 QUERY ───┤                                                ├──> FUSION ──> final ranking ──> (rerank) ──> top results
          └──> vector search (kNN)    ──> ranked list B ──┘
```

### Reciprocal Rank Fusion (RRF): the standard way to merge

Keyword scores (BM25) and vector scores (cosine) are on **different scales** and can't be added directly. RRF sidesteps this by using only **rank positions**:

```plaintext
RRF score(doc) = SUM over each list of   1 / (k + rank_in_that_list)        (k is commonly 60)

Doc X: rank 1 in keyword list, rank 3 in vector list  ->  1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
Doc Y: rank 2 in keyword list, absent from vector list ->  1/(60+2)           = 0.0161
Doc X wins: it did well in BOTH lists.
```

Documents that appear high in *both* lists rise to the top; a document strong in only one still gets credit. RRF needs no score calibration and is robust, which is why Azure AI Search and many other systems use it.

```csharp
static List<(string Id, double Score)> ReciprocalRankFusion(
    IEnumerable<IReadOnlyList<string>> rankedLists, int k = 60)
{
    var scores = new Dictionary<string, double>();

    foreach (var list in rankedLists)
        for (int rank = 0; rank < list.Count; rank++)
            scores[list[rank]] = scores.GetValueOrDefault(list[rank]) + 1.0 / (k + rank + 1);

    return scores.OrderByDescending(kv => kv.Value)
                 .Select(kv => (kv.Key, kv.Value))
                 .ToList();
}

var fused = ReciprocalRankFusion(new[] { keywordIds, vectorIds }).Take(10).ToList();
```

An alternative is **weighted score fusion** (normalize each list's scores, then take a weighted sum) when you want to favor one signal — but it requires careful normalization and tuning.

### Hybrid search on each platform

```plaintext
AZURE AI SEARCH   Built in: supply a text query AND a vector query in one request; results are
                  merged with RRF automatically (Section 7). Add the semantic ranker for reranking.

POSTGRESQL        Combine pgvector with PostgreSQL's native full-text search (tsvector / tsquery),
                  and fuse in SQL (below) or in application code.

REDIS             Use TEXT / TAG / NUMERIC filters as pre-filters on a KNN query, or run a text
                  query and a KNN query and fuse in application code. (Check whether your
                  Redis version offers a built-in hybrid/fusion command.)

MONGODB ATLAS     Run an Atlas Search (full-text) query and a $vectorSearch and fuse them — newer
                  Atlas versions offer built-in rank-fusion stages; otherwise fuse in the
                  application.
```

### Hybrid search in PostgreSQL: a complete RRF query

```sql
-- One-time: a generated tsvector column and a GIN index for fast full-text search
ALTER TABLE documents
  ADD COLUMN content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX documents_content_tsv_idx ON documents USING gin (content_tsv);
```

```sql
-- $1 = query embedding (vector), $2 = query text
WITH semantic AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
    FROM documents
    ORDER BY embedding <=> $1
    LIMIT 20
),
keyword AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', $2)) DESC) AS rank
    FROM documents
    WHERE content_tsv @@ plainto_tsquery('english', $2)
    ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', $2)) DESC
    LIMIT 20
)
SELECT COALESCE(s.id, k.id) AS id,
       COALESCE(1.0 / (60 + s.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS rrf_score
FROM semantic s
FULL OUTER JOIN keyword k ON s.id = k.id
ORDER BY rrf_score DESC
LIMIT 10;
```

### Add a reranker for the final polish

```plaintext
Pipeline used by many production RAG systems:

  1. Hybrid retrieval of a broad candidate set (top 20-50)       <- cheap, high recall
  2. RERANK with a cross-encoder or semantic ranker               <- slower, high precision
  3. Keep the top 3-8 passages for the LLM prompt                 <- small, focused context

A reranker reads the query and each passage together, so it judges relevance far better than
vector distance alone — and running it only on a short candidate list keeps it affordable.
```

---

## 12. Choosing a Vector Store

### A side-by-side view

| | Azure AI Search | PostgreSQL + pgvector | Redis | MongoDB Atlas |
|---|---|---|---|---|
| **Best at** | Managed hybrid + semantic ranking | Vectors next to relational data | Lowest latency, caching | Vectors inside documents |
| **Hybrid search** | Built in (RRF) | DIY (FTS + RRF) | Filters + DIY fusion | Atlas Search + fusion |
| **Filtering** | OData, pre-filter by default | SQL `WHERE` (ANN-filter caveat) | TAG/NUMERIC pre-filter | `filter` fields in index |
| **Operations** | Managed service | Your DB (or managed Postgres) | Managed or self-run | Managed (Atlas) |
| **Cost shape** | Per service tier/unit | Part of your DB cost | Memory-driven | Dedicated search nodes |
| **Watch out for** | Baseline cost; sync with source | Filtered ANN recall; index memory | Memory cost; persistence config | Atlas-centric; filter field setup |

### A decision guide

```plaintext
Already on PostgreSQL, moderate scale, want transactions and SQL joins   -> pgvector
Azure shop that needs strong hybrid search and ranking with little code  -> Azure AI Search
Need single-digit-millisecond lookups, semantic cache, or session memory -> Redis
Data already modeled as documents in MongoDB Atlas                        -> MongoDB vector search
Hundreds of millions of vectors or specialized vector features            -> a dedicated vector database
```

### Whatever you choose, keep the system of record separate in your head

```plaintext
The vector store is an INDEX over your content, not the content's home. Keep the authoritative
data in your primary database or storage, keep the vector index rebuildable from it, and plan
how updates, deletes, and re-embedding flow from the source to the index. An index you can't
rebuild is a liability.
```

---

## 13. Putting It Together: An Ingestion and Query Pipeline

### Two flows share one embedding model and one index

```plaintext
INGESTION (offline / on content change)
  source documents
    -> clean + split into chunks (a few hundred tokens, light overlap, keep title/section as context)
    -> embed each chunk (batched, rate-limited)
    -> upsert into the vector store:  id, chunk text, vector, metadata
                                       (source, position, tenant/ACL, embedding-model name, timestamp)

QUERY (online, per request)
  user question
    -> embed the question (SAME model)
    -> hybrid retrieval (vector + keyword) with filters (tenant, permissions)  -> top 20-50 candidates
    -> rerank -> top 3-8 passages
    -> build the LLM prompt: instructions + passages + question
    -> generate answer with citations
    -> validate that cited passages really exist in the retrieved set
```

### The skeleton in C#

```csharp
public async Task IngestAsync(IEnumerable<SourceDoc> docs, CancellationToken ct)
{
    foreach (var doc in docs)
    {
        var chunks = chunker.Split(doc.Text, maxTokens: 400, overlapTokens: 50)
                            .Select(c => $"{doc.Title}\n\n{c}")             // carry context into each chunk
                            .ToList();

        var embeddings = (await embeddingClient.GenerateEmbeddingsAsync(chunks, cancellationToken: ct)).Value;

        var records = chunks.Select((text, i) => new ChunkRecord(
            Id: $"{doc.Id}:{i}", DocId: doc.Id, Text: text,
            Vector: embeddings[i].ToFloats(),
            TenantId: doc.TenantId, Model: "text-embedding-3-small"));

        await store.UpsertAsync(records, ct);        // delete old chunks of doc.Id first if the doc changed
    }
}

public async Task<List<ChunkRecord>> RetrieveAsync(string question, int tenantId, CancellationToken ct)
{
    var queryVector = (await embeddingClient.GenerateEmbeddingAsync(question, cancellationToken: ct)).Value.ToFloats();

    var candidates = await store.HybridSearchAsync(question, queryVector, tenantId, top: 30, ct);   // filter by tenant inside the query
    return await reranker.RerankAsync(question, candidates, keep: 5, ct);
}
```

### Evaluate retrieval separately from generation

```plaintext
If the answer is wrong, find out WHICH half failed: retrieval (the right passage wasn't found)
or generation (it was found but misused). Measure retrieval directly:

  1. Build a test set: realistic questions, each paired with the passage(s) that answer it.
  2. For each question, run retrieval and record where the correct passage ranked.
  3. Report:
       Recall@k  = fraction of questions whose correct passage appears in the top k
       MRR       = average of 1 / (rank of the first correct passage)
  4. Change one thing at a time (chunk size, model, hybrid on/off, reranker) and re-measure.
```

---

## 14. Common Pitfalls

| Pitfall | Why it hurts | Better approach |
|---|---|---|
| Embedding queries and documents with different models | Vectors from different models live in different spaces; similarity is meaningless | Use one model for both, and store the model name with every vector (Section 2) |
| Changing embedding models without re-embedding | Old and new vectors are incomparable; results silently degrade | Version the model; re-embed the full corpus (or run old and new indexes side by side) when switching (Section 2) |
| Embedding whole documents as one vector | Meaning blurs; input limits are hit; retrieval is imprecise | Chunk into focused passages with light overlap and keep source references (Section 2) |
| Treating cosine *distance* as similarity (or vice versa) | Ranking comes out exactly backwards | Know each system's convention: pgvector and Redis return distances, MongoDB returns similarity (Sections 4, 8–10) |
| Index metric differs from query metric | Index isn't used (full scan) or results are wrong | Create the index with the metric you query with, e.g. `vector_cosine_ops` with `<=>` (Section 8) |
| Mismatched dimension counts | Inserts or queries fail, or the field can't be indexed | Match index dimensions to the model; shorten embeddings or use a lower-precision type if you exceed index limits (Sections 2, 8) |
| Post-filtering ANN results (pgvector) | A selective filter leaves fewer than k rows, or none | Raise `ef_search`, use iterative scans, partial indexes, or partitioning (Section 8) |
| Forgetting tenant/permission filters | One user's query can retrieve another user's documents — a data breach | Filter by tenant/ACL inside the vector query, never after retrieval; test it explicitly (Sections 5, 13) |
| Pure vector search for exact-match needs | Embeddings are weak on IDs, codes, and rare terms | Use hybrid search with keyword retrieval (Section 11) |
| Adding BM25 and cosine scores directly | The scales are incomparable | Fuse by rank with RRF, or normalize carefully before weighting (Section 11) |
| Returning the top-k regardless of quality | The "least bad" result is presented as an answer | Apply a score threshold tuned on real queries, and let the system say "no match" (Section 5) |
| Never measuring retrieval quality | You can't tell whether tuning helped, or whether retrieval or generation is failing | Build a labeled question set; track Recall@k and MRR (Section 13) |
| Stale vectors after source updates | Search returns results for text that no longer exists | Re-embed on change; delete old chunks when a document is updated or removed (Sections 10, 13) |
| Over-tuning ANN for speed without measuring recall | Quietly missing relevant results | Compare ANN results to exact search on sample queries; tune `ef_search`/`numCandidates`/probes against recall (Section 5) |
| Adopting a dedicated vector database too early | Another system to run, secure, and keep in sync | Start with the database you already operate; migrate only when scale or features demand (Section 6) |
| Ignoring that embeddings are derived from sensitive data | Vectors can leak information about the source text | Apply the same access control, encryption, and retention as the source (Section 2) |

---

## Quick Reference Table

| Concept | Key idea | Practical takeaway |
|---|---|---|
| Embedding | A vector of numbers representing meaning | Similar items → nearby points |
| Text embedding model | Encodes text into a fixed-length vector | Same model for documents and queries; record its version |
| Dimensions | Length of the vector (e.g. 1536) | Must match across model, index, and queries |
| Chunking | Split documents into focused passages | A few hundred tokens, light overlap, keep context |
| Semantic similarity | Closeness in meaning, not wording | Great for paraphrase; weak on exact tokens and numbers |
| Cosine similarity | Angle between vectors (-1 to 1) | Higher = more similar; ignores vector length |
| Cosine distance | `1 - cosine similarity` | Lower = more similar; what many databases return |
| Dot product | `sum(A[i]*B[i])` | Same ranking as cosine on unit-length vectors |
| Exact kNN | Compare against every vector | Perfect recall; fine for small collections |
| ANN (HNSW / IVF) | Approximate nearest neighbors via an index | Fast at scale; tune recall against speed |
| Pre-filtering | Apply filters during the vector search | Needed for tenancy and permissions |
| Reranking | Second-stage relevance scoring | Retrieve broad, rerank narrow |
| Vector database | Stores vectors with ANN indexes and filters | Prefer the database you already run until scale demands more |
| Azure AI Search | Managed vector + keyword + semantic ranking | Built-in hybrid with RRF |
| pgvector | Vector type and indexes in PostgreSQL | `<=>` operator + HNSW/IVFFlat; mind filtered-ANN recall |
| Redis vector search | In-memory KNN via `FT.SEARCH` | Lowest latency; ideal for semantic caching |
| MongoDB `$vectorSearch` | ANN search over document fields | Declare `filter` fields; tune `numCandidates` |
| Hybrid search | Keyword + vector retrieval, fused | Beats either alone on mixed queries |
| RRF | `sum(1 / (k + rank))`, k ≈ 60 | Merges ranked lists without score calibration |
| Recall@k / MRR | Retrieval quality metrics | Measure retrieval separately from generation |

---

## Conclusion

Embeddings turn a fuzzy problem — "find content that *means* something similar" — into a precise one: find the nearest points in a vector space. Everything practical follows from a few ideas. Embeddings are only comparable when produced by the same model, so the model becomes part of your data's identity. Cosine similarity (or dot product on normalized vectors) is the standard yardstick, with the constant risk of confusing a distance with a similarity. Exact search is simple and perfect but linear, so at scale you accept approximate indexes like HNSW and tune recall against speed deliberately. And retrieval quality depends as much on the unglamorous parts — chunking, filtering, thresholds, and evaluation — as on the model.

The platform choice matters less than it first appears, because the same patterns recur everywhere: define an index with the right dimensions and metric, store vectors beside their metadata, query with filters, and measure what comes back. Azure AI Search gives you managed hybrid retrieval and ranking; pgvector brings vectors into the relational database you already trust; Redis offers memory-speed lookups and semantic caching; MongoDB keeps embeddings inside your documents. Start with the one closest to your existing data, add keyword retrieval and fusion because real queries mix meaning with exact terms, put a reranker on top when precision matters, and always enforce tenant and permission filters inside the query. Build that foundation and measure it with Recall@k, and the retrieval layer of your RAG applications stops being a black box and becomes something you can improve on purpose.

---

*Found this useful? Feel free to star the repo, open an issue with corrections, or share the "the chatbot found a perfect paraphrase but missed the exact error code" story that made the case for hybrid search click better than any benchmark.*
