import * as SecureStore from 'expo-secure-store';

import type { AppConfig, Planner } from './types';

const SERVER_URL_KEY = 'campus-agent.server-url';
const ACCESS_TOKEN_KEY = 'campus-agent.access-token';
const PLANNER_KEY = 'campus-agent.planner';
const SESSION_ID_KEY = 'campus-agent.session-id';

export async function loadConfig(): Promise<AppConfig | null> {
  const [baseUrl, token, storedPlanner] = await Promise.all([
    SecureStore.getItemAsync(SERVER_URL_KEY),
    SecureStore.getItemAsync(ACCESS_TOKEN_KEY),
    SecureStore.getItemAsync(PLANNER_KEY),
  ]);
  if (!baseUrl || !token) {
    return null;
  }
  const planner: Planner = storedPlanner === 'deepseek' ? 'deepseek' : 'rule';
  return { baseUrl, token, planner };
}

export async function saveConfig(config: AppConfig): Promise<void> {
  await Promise.all([
    SecureStore.setItemAsync(SERVER_URL_KEY, config.baseUrl),
    SecureStore.setItemAsync(ACCESS_TOKEN_KEY, config.token),
    SecureStore.setItemAsync(PLANNER_KEY, config.planner),
  ]);
}

function createSessionId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).slice(2, 14);
  return `mobile_${timestamp}_${random}`;
}

export async function loadOrCreateSessionId(): Promise<string> {
  const existing = await SecureStore.getItemAsync(SESSION_ID_KEY);
  if (existing) {
    return existing;
  }
  const created = createSessionId();
  await SecureStore.setItemAsync(SESSION_ID_KEY, created);
  return created;
}
