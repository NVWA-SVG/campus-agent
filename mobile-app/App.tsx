import { StatusBar } from 'expo-status-bar';
import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import {
  ApiError,
  checkHealth,
  clearHistory,
  fetchHistory,
  messageForError,
  streamChat,
} from './src/api';
import { MessageBubble } from './src/components/MessageBubble';
import { SettingsModal } from './src/components/SettingsModal';
import { normalizeServerUrl, validateAccessToken } from './src/config';
import { loadConfig, loadOrCreateSessionId, saveConfig } from './src/storage';
import type { AppConfig, ChatMessage, Citation, Planner } from './src/types';

type ConnectionStatus = 'checking' | 'online' | 'offline';

const QUICK_QUESTIONS = [
  '周三有什么课程？',
  '校园卡丢了怎么补办？',
  '图书馆借阅规则是什么？',
];

function newMessageId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function citationSources(citations: Citation[]): string[] {
  return Array.from(
    new Set(
      citations.flatMap((citation) => {
        const source = citation.source || citation.title;
        return source ? [source] : [];
      }),
    ),
  );
}

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [sessionId, setSessionId] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [trace, setTrace] = useState('');
  const [error, setError] = useState('');
  const [connection, setConnection] = useState<ConnectionStatus>('checking');
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsError, setSettingsError] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<FlatList<ChatMessage>>(null);

  useEffect(() => {
    let cancelled = false;

    const initialize = async () => {
      try {
        const [storedConfig, storedSessionId] = await Promise.all([
          loadConfig(),
          loadOrCreateSessionId(),
        ]);
        if (cancelled) {
          return;
        }
        setSessionId(storedSessionId);
        if (!storedConfig) {
          setConnection('offline');
          setSettingsVisible(true);
          return;
        }

        setConfig(storedConfig);
        try {
          const history = await fetchHistory(storedConfig, storedSessionId);
          if (!cancelled) {
            setMessages(history);
            setConnection('online');
          }
        } catch (loadError) {
          if (!cancelled) {
            setConnection('offline');
            setError(messageForError(loadError));
          }
        }
      } catch {
        if (!cancelled) {
          setConnection('offline');
          setError('无法读取本机安全存储，请重新配置连接');
          setSettingsVisible(true);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void initialize();
    return () => {
      cancelled = true;
      abortRef.current?.abort();
    };
  }, []);

  const handleSaveSettings = async (
    baseUrl: string,
    token: string,
    planner: Planner,
  ) => {
    setSettingsError('');
    setSettingsSaving(true);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8_000);

    try {
      const candidate: AppConfig = {
        baseUrl: normalizeServerUrl(baseUrl),
        token: validateAccessToken(token),
        planner,
      };
      await checkHealth(candidate.baseUrl, controller.signal);
      const history = await fetchHistory(candidate, sessionId, controller.signal);
      await saveConfig(candidate);
      setConfig(candidate);
      setMessages(history);
      setConnection('online');
      setError('');
      setSettingsVisible(false);
    } catch (saveError) {
      setConnection('offline');
      setSettingsError(messageForError(saveError));
    } finally {
      clearTimeout(timeout);
      setSettingsSaving(false);
    }
  };

  const handleRefresh = async () => {
    if (!config || !sessionId || streaming) {
      return;
    }
    setConnection('checking');
    setError('');
    try {
      const history = await fetchHistory(config, sessionId);
      setMessages(history);
      setConnection('online');
    } catch (refreshError) {
      setConnection('offline');
      setError(messageForError(refreshError));
    }
  };

  const handlePlannerChange = async (planner: Planner) => {
    if (!config || planner === config.planner || streaming) {
      return;
    }
    const updated = { ...config, planner };
    setConfig(updated);
    try {
      await saveConfig(updated);
    } catch {
      setError('规划器设置未能保存到本机');
    }
  };

  const handleSend = async () => {
    const query = input.trim();
    if (!query || !config || !sessionId || streaming) {
      if (!config) {
        setSettingsVisible(true);
      }
      return;
    }

    const userMessage: ChatMessage = {
      id: newMessageId('user'),
      role: 'user',
      content: query,
    };
    setMessages((current) => [...current, userMessage]);
    setInput('');
    setError('');
    setTrace('正在连接 Agent…');
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const result = await streamChat(
        config,
        sessionId,
        query,
        { onTrace: setTrace },
        controller.signal,
      );
      setMessages((current) => [
        ...current,
        {
          id: newMessageId('assistant'),
          role: 'assistant',
          content: result.answer,
          sources: citationSources(result.citations),
        },
      ]);
      setConnection('online');
    } catch (streamError) {
      setError(messageForError(streamError));
      if (streamError instanceof ApiError) {
        setConnection(streamError.status === 401 ? 'offline' : 'online');
      } else if (!(streamError instanceof Error && streamError.name === 'AbortError')) {
        setConnection('offline');
      }
    } finally {
      abortRef.current = null;
      setStreaming(false);
      setTrace('');
    }
  };

  const handleClear = () => {
    if (!config || !sessionId || streaming) {
      return;
    }
    Alert.alert('清空当前会话？', '服务器中的这段对话历史也会被删除。', [
      { text: '取消', style: 'cancel' },
      {
        text: '清空',
        style: 'destructive',
        onPress: () => {
          void (async () => {
            try {
              await clearHistory(config, sessionId);
              setMessages([]);
              setError('');
            } catch (clearError) {
              setError(messageForError(clearError));
            }
          })();
        },
      },
    ]);
  };

  if (loading) {
    return (
      <SafeAreaView style={styles.loadingScreen}>
        <StatusBar style="dark" />
        <ActivityIndicator color="#1E5AA8" size="large" />
        <Text style={styles.loadingText}>正在读取本机配置…</Text>
      </SafeAreaView>
    );
  }

  const statusText =
    connection === 'online' ? '服务器在线' : connection === 'checking' ? '正在同步' : '连接异常';

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <View style={styles.brandBlock}>
          <Text style={styles.brand}>校园助手</Text>
          <Pressable onPress={() => void handleRefresh()} disabled={!config || streaming}>
            <View style={styles.statusRow}>
              <View
                style={[
                  styles.statusDot,
                  connection === 'online'
                    ? styles.onlineDot
                    : connection === 'checking'
                      ? styles.checkingDot
                      : styles.offlineDot,
                ]}
              />
              <Text style={styles.statusText}>{statusText}</Text>
            </View>
          </Pressable>
        </View>
        <View style={styles.headerActions}>
          {messages.length > 0 ? (
            <Pressable onPress={handleClear} disabled={streaming} hitSlop={8}>
              <Text style={styles.headerAction}>清空</Text>
            </Pressable>
          ) : null}
          <Pressable
            onPress={() => {
              setSettingsError('');
              setSettingsVisible(true);
            }}
            disabled={streaming}
            hitSlop={8}
          >
            <Text style={styles.headerAction}>设置</Text>
          </Pressable>
        </View>
      </View>

      <View style={styles.plannerBar}>
        <Text style={styles.plannerLabel}>规划器</Text>
        {(['rule', 'deepseek'] as const).map((planner) => (
          <Pressable
            key={planner}
            disabled={!config || streaming}
            onPress={() => void handlePlannerChange(planner)}
            style={[
              styles.plannerChip,
              config?.planner === planner && styles.plannerChipActive,
            ]}
          >
            <Text
              style={[
                styles.plannerChipText,
                config?.planner === planner && styles.plannerChipTextActive,
              ]}
            >
              {planner === 'rule' ? 'Rule' : 'DeepSeek'}
            </Text>
          </Pressable>
        ))}
      </View>

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 8 : 0}
        style={styles.flex}
      >
        <FlatList
          ref={listRef}
          contentContainerStyle={messages.length === 0 ? styles.emptyList : styles.messageList}
          data={messages}
          keyExtractor={(message) => message.id}
          keyboardDismissMode="interactive"
          keyboardShouldPersistTaps="handled"
          ListEmptyComponent={
            <View style={styles.welcomeCard}>
              <Text style={styles.welcomeTitle}>今天想了解什么？</Text>
              <Text style={styles.welcomeText}>
                可查询课程、校园办事流程和本地知识库内容。
              </Text>
              <View style={styles.quickList}>
                {QUICK_QUESTIONS.map((question) => (
                  <Pressable
                    key={question}
                    onPress={() => setInput(question)}
                    style={styles.quickButton}
                  >
                    <Text style={styles.quickText}>{question}</Text>
                  </Pressable>
                ))}
              </View>
            </View>
          }
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
          renderItem={({ item }) => <MessageBubble message={item} />}
        />

        {trace ? (
          <View style={styles.traceBar}>
            <ActivityIndicator color="#1E5AA8" size="small" />
            <Text numberOfLines={1} style={styles.traceText}>
              {trace}
            </Text>
          </View>
        ) : null}

        {error ? (
          <Pressable onPress={() => setError('')} style={styles.errorBar}>
            <Text style={styles.errorText}>{error}</Text>
            <Text style={styles.dismissError}>×</Text>
          </Pressable>
        ) : null}

        <View style={styles.composer}>
          <TextInput
            editable={!streaming}
            maxLength={2000}
            multiline
            onChangeText={setInput}
            placeholder={config ? '输入校园事务问题…' : '请先配置服务器'}
            placeholderTextColor="#8794A8"
            style={styles.messageInput}
            value={input}
          />
          <Pressable
            accessibilityLabel={streaming ? '停止生成' : '发送消息'}
            disabled={!streaming && (!input.trim() || !config)}
            onPress={streaming ? () => abortRef.current?.abort() : () => void handleSend()}
            style={({ pressed }) => [
              styles.sendButton,
              !streaming && (!input.trim() || !config) && styles.sendButtonDisabled,
              pressed && styles.sendButtonPressed,
            ]}
          >
            <Text style={styles.sendText}>{streaming ? '停止' : '发送'}</Text>
          </Pressable>
        </View>
      </KeyboardAvoidingView>

      <SettingsModal
        error={settingsError}
        initialConfig={config}
        onClose={() => {
          if (config && !settingsSaving) {
            setSettingsVisible(false);
          }
        }}
        onSave={(baseUrl, token, planner) =>
          void handleSaveSettings(baseUrl, token, planner)
        }
        saving={settingsSaving}
        visible={settingsVisible}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  safeArea: { backgroundColor: '#F3F6FA', flex: 1 },
  loadingScreen: {
    alignItems: 'center',
    backgroundColor: '#F3F6FA',
    flex: 1,
    justifyContent: 'center',
  },
  loadingText: { color: '#607089', fontSize: 14, marginTop: 12 },
  header: {
    alignItems: 'center',
    backgroundColor: '#153E70',
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 18,
    paddingVertical: 13,
  },
  brandBlock: { gap: 3 },
  brand: { color: '#FFFFFF', fontSize: 21, fontWeight: '800', letterSpacing: 0.3 },
  statusRow: { alignItems: 'center', flexDirection: 'row', gap: 6 },
  statusDot: { borderRadius: 4, height: 7, width: 7 },
  onlineDot: { backgroundColor: '#58D69B' },
  checkingDot: { backgroundColor: '#F5C563' },
  offlineDot: { backgroundColor: '#FF8C8C' },
  statusText: { color: '#D6E3F5', fontSize: 12 },
  headerActions: { alignItems: 'center', flexDirection: 'row', gap: 18 },
  headerAction: { color: '#FFFFFF', fontSize: 14, fontWeight: '600' },
  plannerBar: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderBottomColor: '#DEE5EF',
    borderBottomWidth: StyleSheet.hairlineWidth,
    flexDirection: 'row',
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 9,
  },
  plannerLabel: { color: '#738197', fontSize: 12, marginRight: 2 },
  plannerChip: {
    backgroundColor: '#F1F4F8',
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 5,
  },
  plannerChipActive: { backgroundColor: '#DDEBFF' },
  plannerChipText: { color: '#68778D', fontSize: 12, fontWeight: '600' },
  plannerChipTextActive: { color: '#174C8D' },
  messageList: { paddingBottom: 10, paddingTop: 11 },
  emptyList: { flexGrow: 1, justifyContent: 'center', padding: 22 },
  welcomeCard: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E2E8F0',
    borderRadius: 20,
    borderWidth: StyleSheet.hairlineWidth,
    padding: 22,
  },
  welcomeTitle: { color: '#172033', fontSize: 22, fontWeight: '800' },
  welcomeText: { color: '#607089', fontSize: 14, lineHeight: 21, marginTop: 8 },
  quickList: { gap: 9, marginTop: 20 },
  quickButton: {
    backgroundColor: '#F2F6FC',
    borderColor: '#DBE5F2',
    borderRadius: 12,
    borderWidth: 1,
    paddingHorizontal: 13,
    paddingVertical: 11,
  },
  quickText: { color: '#24568F', fontSize: 14, lineHeight: 20 },
  traceBar: {
    alignItems: 'center',
    backgroundColor: '#EAF2FF',
    flexDirection: 'row',
    gap: 9,
    paddingHorizontal: 17,
    paddingVertical: 9,
  },
  traceText: { color: '#315C8F', flex: 1, fontSize: 12 },
  errorBar: {
    alignItems: 'center',
    backgroundColor: '#FFF0F0',
    flexDirection: 'row',
    gap: 10,
    paddingHorizontal: 16,
    paddingVertical: 9,
  },
  errorText: { color: '#983838', flex: 1, fontSize: 12, lineHeight: 18 },
  dismissError: { color: '#983838', fontSize: 20, lineHeight: 20 },
  composer: {
    alignItems: 'flex-end',
    backgroundColor: '#FFFFFF',
    borderTopColor: '#DEE5EF',
    borderTopWidth: StyleSheet.hairlineWidth,
    flexDirection: 'row',
    gap: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  messageInput: {
    backgroundColor: '#F3F6FA',
    borderColor: '#D8E0EB',
    borderRadius: 18,
    borderWidth: 1,
    color: '#172033',
    flex: 1,
    fontSize: 16,
    lineHeight: 22,
    maxHeight: 120,
    minHeight: 44,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  sendButton: {
    alignItems: 'center',
    backgroundColor: '#1E5AA8',
    borderRadius: 16,
    height: 44,
    justifyContent: 'center',
    minWidth: 62,
    paddingHorizontal: 13,
  },
  sendButtonDisabled: { backgroundColor: '#ABB7C7' },
  sendButtonPressed: { opacity: 0.78 },
  sendText: { color: '#FFFFFF', fontSize: 14, fontWeight: '700' },
});
