import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import type { AppConfig, Planner } from '../types';

interface SettingsModalProps {
  visible: boolean;
  initialConfig: AppConfig | null;
  saving: boolean;
  error: string;
  onClose: () => void;
  onSave: (baseUrl: string, token: string, planner: Planner) => void;
}

export function SettingsModal({
  visible,
  initialConfig,
  saving,
  error,
  onClose,
  onSave,
}: SettingsModalProps) {
  const [baseUrl, setBaseUrl] = useState('');
  const [token, setToken] = useState('');
  const [planner, setPlanner] = useState<Planner>('rule');
  const [showToken, setShowToken] = useState(false);

  useEffect(() => {
    if (visible) {
      setBaseUrl(initialConfig?.baseUrl ?? '');
      setToken(initialConfig?.token ?? '');
      setPlanner(initialConfig?.planner ?? 'rule');
      setShowToken(false);
    }
  }, [initialConfig, visible]);

  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onClose}>
      <SafeAreaView style={styles.safeArea}>
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          style={styles.flex}
        >
          <View style={styles.header}>
            <View>
              <Text style={styles.title}>连接设置</Text>
              <Text style={styles.subtitle}>配置只保存在本机安全存储中</Text>
            </View>
            {initialConfig ? (
              <Pressable onPress={onClose} disabled={saving} hitSlop={12}>
                <Text style={styles.close}>关闭</Text>
              </Pressable>
            ) : null}
          </View>

          <ScrollView contentContainerStyle={styles.form} keyboardShouldPersistTaps="handled">
            <Text style={styles.label}>服务器地址</Text>
            <TextInput
              autoCapitalize="none"
              autoCorrect={false}
              editable={!saving}
              keyboardType="url"
              onChangeText={setBaseUrl}
              placeholder="http://192.168.1.8:8000"
              style={styles.input}
              value={baseUrl}
            />
            <Text style={styles.help}>
              手机需填写电脑的局域网 IP；公网地址必须使用 HTTPS。
            </Text>

            <Text style={styles.label}>访问 Token</Text>
            <View style={styles.tokenRow}>
              <TextInput
                autoCapitalize="none"
                autoCorrect={false}
                editable={!saving}
                onChangeText={setToken}
                placeholder="服务器的 CAMPUS_MOBILE_API_TOKEN"
                secureTextEntry={!showToken}
                style={[styles.input, styles.tokenInput]}
                value={token}
              />
              <Pressable
                accessibilityRole="button"
                onPress={() => setShowToken((current) => !current)}
                style={styles.revealButton}
              >
                <Text style={styles.revealText}>{showToken ? '隐藏' : '显示'}</Text>
              </Pressable>
            </View>

            <Text style={styles.label}>默认规划器</Text>
            <View style={styles.plannerRow}>
              {(['rule', 'deepseek'] as const).map((value) => (
                <Pressable
                  key={value}
                  disabled={saving}
                  onPress={() => setPlanner(value)}
                  style={[styles.plannerButton, planner === value && styles.plannerActive]}
                >
                  <Text
                    style={[
                      styles.plannerText,
                      planner === value && styles.plannerTextActive,
                    ]}
                  >
                    {value === 'rule' ? 'Rule（本地）' : 'DeepSeek'}
                  </Text>
                </Pressable>
              ))}
            </View>
            <Text style={styles.help}>
              DeepSeek Key 只配置在服务器；App 不保存模型 Key。
            </Text>

            {error ? <Text style={styles.error}>{error}</Text> : null}

            <Pressable
              disabled={saving}
              onPress={() => onSave(baseUrl, token, planner)}
              style={[styles.saveButton, saving && styles.disabledButton]}
            >
              {saving ? (
                <ActivityIndicator color="#FFFFFF" />
              ) : (
                <Text style={styles.saveText}>验证并保存</Text>
              )}
            </Pressable>
          </ScrollView>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F6F8FC' },
  flex: { flex: 1 },
  header: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderBottomColor: '#E2E8F0',
    borderBottomWidth: StyleSheet.hairlineWidth,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 18,
  },
  title: { color: '#172033', fontSize: 22, fontWeight: '700' },
  subtitle: { color: '#6A788E', fontSize: 13, marginTop: 3 },
  close: { color: '#1E5AA8', fontSize: 16, fontWeight: '600' },
  form: { padding: 20, paddingBottom: 42 },
  label: { color: '#28364D', fontSize: 14, fontWeight: '700', marginBottom: 7, marginTop: 16 },
  input: {
    backgroundColor: '#FFFFFF',
    borderColor: '#CDD7E5',
    borderRadius: 12,
    borderWidth: 1,
    color: '#172033',
    fontSize: 16,
    minHeight: 50,
    paddingHorizontal: 13,
  },
  help: { color: '#6A788E', fontSize: 12, lineHeight: 18, marginTop: 6 },
  tokenRow: { alignItems: 'center', flexDirection: 'row' },
  tokenInput: { flex: 1, paddingRight: 64 },
  revealButton: { marginLeft: -58, paddingHorizontal: 10, paddingVertical: 10 },
  revealText: { color: '#1E5AA8', fontSize: 13, fontWeight: '600' },
  plannerRow: { flexDirection: 'row', gap: 10 },
  plannerButton: {
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    borderColor: '#CDD7E5',
    borderRadius: 12,
    borderWidth: 1,
    flex: 1,
    paddingVertical: 13,
  },
  plannerActive: { backgroundColor: '#EAF2FF', borderColor: '#1E5AA8' },
  plannerText: { color: '#52637A', fontSize: 14, fontWeight: '600' },
  plannerTextActive: { color: '#174C8D' },
  error: {
    backgroundColor: '#FFF0F0',
    borderRadius: 10,
    color: '#A33131',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 18,
    padding: 12,
  },
  saveButton: {
    alignItems: 'center',
    backgroundColor: '#1E5AA8',
    borderRadius: 13,
    height: 52,
    justifyContent: 'center',
    marginTop: 24,
  },
  disabledButton: { opacity: 0.55 },
  saveText: { color: '#FFFFFF', fontSize: 16, fontWeight: '700' },
});
