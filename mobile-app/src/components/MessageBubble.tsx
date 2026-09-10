import { StyleSheet, Text, View } from 'react-native';

import type { ChatMessage } from '../types';

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  return (
    <View style={[styles.row, isUser ? styles.userRow : styles.assistantRow]}>
      <View style={[styles.bubble, isUser ? styles.userBubble : styles.assistantBubble]}>
        <Text style={[styles.content, isUser && styles.userContent]} selectable>
          {message.content}
        </Text>
        {message.sources && message.sources.length > 0 ? (
          <View style={styles.sources}>
            <Text style={styles.sourcesLabel}>参考来源</Text>
            <Text style={styles.sourcesText}>{message.sources.join(' · ')}</Text>
          </View>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    paddingHorizontal: 16,
    paddingVertical: 5,
  },
  userRow: {
    alignItems: 'flex-end',
  },
  assistantRow: {
    alignItems: 'flex-start',
  },
  bubble: {
    maxWidth: '86%',
    borderRadius: 18,
    paddingHorizontal: 15,
    paddingVertical: 11,
  },
  userBubble: {
    backgroundColor: '#1E5AA8',
    borderBottomRightRadius: 5,
  },
  assistantBubble: {
    backgroundColor: '#FFFFFF',
    borderBottomLeftRadius: 5,
    borderColor: '#E2E8F0',
    borderWidth: StyleSheet.hairlineWidth,
  },
  content: {
    color: '#172033',
    fontSize: 16,
    lineHeight: 24,
  },
  userContent: {
    color: '#FFFFFF',
  },
  sources: {
    borderTopColor: '#E2E8F0',
    borderTopWidth: StyleSheet.hairlineWidth,
    marginTop: 9,
    paddingTop: 7,
  },
  sourcesLabel: {
    color: '#607089',
    fontSize: 11,
    fontWeight: '600',
    marginBottom: 2,
  },
  sourcesText: {
    color: '#52637A',
    fontSize: 12,
    lineHeight: 17,
  },
});
