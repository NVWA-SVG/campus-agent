export type Planner = 'rule' | 'deepseek';

export interface AppConfig {
  baseUrl: string;
  token: string;
  planner: Planner;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
}

export interface Citation {
  source?: string;
  title?: string;
  document_id?: string;
}

export interface ChatResult {
  answer: string;
  citations: Citation[];
}

export interface StreamCallbacks {
  onTrace: (detail: string) => void;
}
