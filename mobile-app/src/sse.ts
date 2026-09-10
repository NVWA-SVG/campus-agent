export interface ParsedSseEvent {
  event: string;
  data: unknown;
}

const MAX_PENDING_EVENT_BYTES = 1024 * 1024;

function parseBlock(block: string): ParsedSseEvent | null {
  let event = 'message';
  const dataLines: string[] = [];

  for (const line of block.split(/\r\n|\r|\n/)) {
    if (!line || line.startsWith(':')) {
      continue;
    }
    const separator = line.indexOf(':');
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? '' : line.slice(separator + 1);
    if (value.startsWith(' ')) {
      value = value.slice(1);
    }
    if (field === 'event') {
      event = value;
    } else if (field === 'data') {
      dataLines.push(value);
    }
  }

  if (dataLines.length === 0) {
    return null;
  }
  const rawData = dataLines.join('\n');
  try {
    return { event, data: JSON.parse(rawData) as unknown };
  } catch {
    throw new Error('服务器返回了无法解析的流式数据');
  }
}

export class SseParser {
  private buffer = '';

  push(chunk: string): ParsedSseEvent[] {
    this.buffer += chunk;
    const events: ParsedSseEvent[] = [];

    while (true) {
      const boundary = /\r?\n\r?\n/.exec(this.buffer);
      if (!boundary || boundary.index === undefined) {
        break;
      }
      const block = this.buffer.slice(0, boundary.index);
      this.buffer = this.buffer.slice(boundary.index + boundary[0].length);
      const parsed = parseBlock(block);
      if (parsed) {
        events.push(parsed);
      }
    }

    if (this.buffer.length > MAX_PENDING_EVENT_BYTES) {
      throw new Error('服务器返回的单个流式事件过大');
    }
    return events;
  }

  finish(): ParsedSseEvent[] {
    if (!this.buffer.trim()) {
      this.buffer = '';
      return [];
    }
    const block = this.buffer;
    this.buffer = '';
    const parsed = parseBlock(block);
    return parsed ? [parsed] : [];
  }
}
