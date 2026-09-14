# ZazGlo

A large-scale product-data platform. This overview focuses on the engineering and scale, not the product surface.

> Project overview only. The source lives in a private repository. Happy to walk through the architecture on request.

## Scale (what I built)

- Multi-source ETL ingesting from **46+ heterogeneous data sources** into a unified catalog of **150k+ entries**, on an automated refresh cadence.
- A 5-stage semantic search system over **157k+ vector embeddings** at **sub-100ms** latency.
- Multi-schema PostgreSQL with **100+ versioned migrations** and GIN/trigram indexing; cut key endpoint latency from seconds to sub-second.
- LLM-powered content generation and enrichment at scale via the Anthropic Claude API.

## Infrastructure

- Full AWS footprint provisioned as code with CDK: ECS Fargate, RDS, Secrets Manager, EventBridge.
- Event-driven pipelines with retry/backoff and zero-downtime deployments.

## Stack

Python, PostgreSQL, AWS (CDK, ECS Fargate, RDS, EventBridge), the Anthropic Claude API and embeddings, plus a Next.js web surface and a native iOS app.

## Status

Built and operated solo under NerdJoy LLC.
