const MIN_TOKEN_LENGTH = 32;
const MAX_TOKEN_LENGTH = 256;

function isPrivateIpv4(hostname: string): boolean {
  const parts = hostname.split('.');
  if (parts.length !== 4) {
    return false;
  }
  const octets = parts.map((part) => Number(part));
  if (octets.some((value) => !Number.isInteger(value) || value < 0 || value > 255)) {
    return false;
  }
  return (
    octets[0] === 10 ||
    (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31) ||
    (octets[0] === 192 && octets[1] === 168) ||
    octets[0] === 127
  );
}

function isLocalHostname(hostname: string): boolean {
  const normalized = hostname.toLowerCase();
  return (
    normalized === 'localhost' ||
    normalized.endsWith('.local') ||
    !normalized.includes('.') ||
    isPrivateIpv4(normalized)
  );
}

export function normalizeServerUrl(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new Error('请输入服务器地址');
  }

  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error('服务器地址格式无效，例如 http://192.168.1.8:8000');
  }

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error('服务器地址只支持 HTTP 或 HTTPS');
  }
  if (parsed.username || parsed.password) {
    throw new Error('服务器地址不能包含用户名或密码');
  }
  if (parsed.pathname !== '/' || parsed.search || parsed.hash) {
    throw new Error('服务器地址只填写协议、主机和端口，不要包含路径');
  }
  if (parsed.protocol === 'http:' && !isLocalHostname(parsed.hostname)) {
    throw new Error('公网服务器必须使用 HTTPS；HTTP 仅允许本机或私有局域网地址');
  }

  return `${parsed.protocol}//${parsed.host}`;
}

export function validateAccessToken(value: string): string {
  if (value.trim() !== value || value.length < MIN_TOKEN_LENGTH || value.length > MAX_TOKEN_LENGTH) {
    throw new Error('访问 Token 必须是32至256位且首尾不含空白');
  }
  return value;
}
