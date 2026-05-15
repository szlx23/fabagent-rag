export type IngestResponse = {
  documents: number;
  chunks: number;
  inserted: number;
  keyword_indexed?: number;
  sources: string[];
};

export type ParsedUploadDocument = {
  source: string;
  text: string;
};

export type ParseUploadResponse = {
  documents: ParsedUploadDocument[];
};

export type ChunkConfig = {
  chunk_size: number;
  chunk_overlap: number;
  min_chunk_size: number;
};

export type AskResponse = {
  question: string;
  intent: "lookup" | "summarize" | "chat";
  query_plan?: {
    original_query?: string;
    rewritten_query?: string;
    expanded_queries?: string[];
    queries?: string[];
    [key: string]: unknown;
  } | null;
  answer: string;
  contexts: Array<{
    source?: string;
    text?: string;
    score?: number;
    vector_score?: number;
    keyword_score?: number;
    retrieval_mode?: string;
    page?: number | null;
    section_title?: string;
    [key: string]: unknown;
  }>;
};
