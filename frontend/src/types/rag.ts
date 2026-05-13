export type IngestResponse = {
  documents: number;
  chunks: number;
  inserted: number;
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
  answer: string;
  contexts: Array<{
    source?: string;
    text?: string;
    score?: number;
    chunk_index?: number;
    [key: string]: unknown;
  }>;
};
