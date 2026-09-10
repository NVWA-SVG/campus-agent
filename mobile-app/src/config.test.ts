import { normalizeServerUrl, validateAccessToken } from './config';

describe('connection configuration', () => {
  test('normalizes HTTPS and private LAN addresses', () => {
    expect(normalizeServerUrl('https://campus.example.com/')).toBe(
      'https://campus.example.com',
    );
    expect(normalizeServerUrl(' http://192.168.1.8:8000 ')).toBe(
      'http://192.168.1.8:8000',
    );
    expect(normalizeServerUrl('http://10.0.2.2:8000')).toBe(
      'http://10.0.2.2:8000',
    );
  });

  test('rejects public cleartext URLs and URLs with a path', () => {
    expect(() => normalizeServerUrl('http://campus.example.com')).toThrow('HTTPS');
    expect(() => normalizeServerUrl('https://campus.example.com/api')).toThrow('路径');
  });

  test('validates shared access token boundaries', () => {
    expect(validateAccessToken('x'.repeat(32))).toHaveLength(32);
    expect(() => validateAccessToken('too-short')).toThrow('32至256位');
    expect(() => validateAccessToken(` ${'x'.repeat(32)}`)).toThrow('空白');
  });
});
