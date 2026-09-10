import { fetch } from 'expo/fetch';

import { SseParser, type ParsedSseEvent } from './sse';
import type {
  AppConfig,
  ChatMessage,
  ChatResult,
  Citation,
  StreamCallbacks,
} from './types';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

function authorizationHeaders(config: AppConfig): Record<string, string> {
  return {
    Accept: 'application/json',
    Authorization: `Bearer ${config.token}`,
  };
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  let detail = '';
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === 'string') {
      detail = payload.detail;
    }
  } catch {
    // 非 JSON 错误页不会暴露给用户。
  }

  if (response.status === 401) {
    return new ApiError('访问 Token 无效，请检查连接设置', 401);
  }
  if (response.status === 429) {
    return new ApiError('服务器当前繁忙，请稍后再试', 429);
  }
  return new ApiError(detail || `服务器请求失败（${response.status}）`, response.status);
}

export async function checkHealth(baseUrl: string, signal?: AbortSignal): Promise<void> {
  const response = await fetch(`${baseUrl}/api/v1/health`, {
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  const payload = (await response.json()) as { status?: unknown };
  if (payload.status !== 'ok') {
    throw new ApiError('服务器健康检查未通过');
  }
}

export async function fetchHistory(
  config: AppConfig,
  sessionId: string,
  signal?: AbortSignal,
): Promise<ChatMessage[]> {
  const response = await fetch(
    `${config.baseUrl}/api/v1/history?session_id=${encodeURIComponent(sessionId)}`,
    { headers: authorizationHeaders(config), signal },
  );
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  const payload = (await response.json()) as { messages?: unknown };
  if (!Array.isArray(payload.messages)) {
    throw new ApiError('服务器返回的历史记录格式无效');
  }

  return payload.messages.flatMap((item, index) => {
    if (
      typeof item !== 'object' ||
      item === null ||
      !('role' in item) ||
      !('content' in item) ||
      (item.role !== 'user' && item.role !== 'assistant') ||
      typeof item.content !== 'string'
    ) {
      return [];
    }
    return [{ id: `history-${index}`, role: item.role, content: item.content }];
  });
}

export async function clearHistory(
  config: AppConfig,
  sessionId: string,
): Promise<void> {
  const response = await fetch(
    `${config.baseUrl}/api/v1/history?session_id=${encodeURIComponent(sessionId)}`,
    { method: 'DELETE', headers: authorizationHeaders(config) },
  );
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null
    ? (value as Record<string, unknown>)
    : null;
}

function consumeEvent(
  parsed: ParsedSseEvent,
  callbacks: StreamCallbacks,
): ChatResult | null {
  const data = asRecord(parsed.data);
  if (parsed.event === 'trace') {
    if (data && typeof data.detail === 'string') {
      callbacks.onTrace(data.detail);
    }
    return null;
  }
  if (parsed.event === 'error') {
    const detail = data && typeof data.detail === 'string' ? data.detail : 'Agent 执行失败';
    throw new ApiError(detail);
  }
  if (parsed.event !== 'result' || !data || typeof data.answer !== 'string') {
    return null;
  }
  const citations = Array.isArray(data.citations)
    ? data.citations.filter(
        (item): item is Citation => typeof item === 'object' && item !== null,
      )
    : [];
  return { answer: data.answer, citations };
}

export async function streamChat(
  config: AppConfig,
  sessionId: string,
  query: string,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<ChatResult> {
  const response = await fetch(`${config.baseUrl}/api/v1/chat/stream`, {
    method: 'POST',
    headers: {
      ...authorizationHeaders(config),
      Accept: 'text/event-stream',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      session_id: sessionId,
      planner: config.planner,
    }),
    signal,
  });
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  if (!response.body) {
    throw new ApiError('当前运行环境不支持流式响应');
  }

  const parser = new SseParser();
  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  let result: ChatResult | null = null;

  const consume = (events: ParsedSseEvent[]) => {
    for (const event of events) {
      const candidate = consumeEvent(event, callbacks);
      if (candidate) {
        result = candidate;
      }
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      consume(parser.push(decoder.decode(value, { stream: true })));
    }
    consume(parser.push(decoder.decode()));
    consume(parser.finish());
  } finally {
    reader.releaseLock();
  }

  if (!result) {
    throw new ApiError('服务器流已结束，但没有返回最终答案');
  }
  return result;
}

export function messageForError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error && error.name === 'AbortError') {
    return '已停止本次生成';
  }
  if (error instanceof Error && error.name !== 'TypeError' && error.message) {
    return error.message;
  }
  return '无法连接服务器，请检查地址、网络和服务状态';
}
