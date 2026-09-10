import { SseParser } from './sse';

describe('SseParser', () => {
  test('parses events split across network chunks', () => {
    const parser = new SseParser();
    expect(parser.push('event: tra')).toEqual([]);
    expect(
      parser.push('ce\ndata: {"detail":"正在规划"}\n\nevent: result\ndata:'),
    ).toEqual([{ event: 'trace', data: { detail: '正在规划' } }]);
    expect(parser.push(' {"answer":"完成"}\n\n')).toEqual([
      { event: 'result', data: { answer: '完成' } },
    ]);
  });

  test('supports CRLF and flushes the final event', () => {
    const parser = new SseParser();
    expect(parser.push('event: done\r\ndata: {"ok":true}\r\n\r\n')).toEqual([
      { event: 'done', data: { ok: true } },
    ]);
    expect(parser.push('event: error\ndata: {"detail":"失败"}')).toEqual([]);
    expect(parser.finish()).toEqual([
      { event: 'error', data: { detail: '失败' } },
    ]);
  });

  test('rejects malformed JSON payloads', () => {
    const parser = new SseParser();
    expect(() => parser.push('event: result\ndata: not-json\n\n')).toThrow(
      '无法解析',
    );
  });
});
