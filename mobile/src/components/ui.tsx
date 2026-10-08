import type { ReactNode } from 'react';
import { Pressable, StyleSheet, Text, View, type ViewStyle } from 'react-native';

import { colors, spacing } from '../theme';

export function Card({ children, style }: { children: ReactNode; style?: ViewStyle }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function Button({
  title,
  onPress,
  variant = 'primary',
  disabled,
}: {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'danger' | 'inverse';
  disabled?: boolean;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        styles.button,
        variant === 'secondary' && styles.buttonSecondary,
        variant === 'danger' && styles.buttonDanger,
        variant === 'inverse' && styles.buttonInverse,
        pressed && variant === 'primary' && { backgroundColor: colors.primaryPressed },
        (pressed || disabled) && variant !== 'primary' && { opacity: 0.7 },
        disabled && { opacity: 0.5 },
      ]}
    >
      <Text style={[styles.buttonText, variant !== 'primary' && styles.buttonTextAlt,
        variant === 'danger' && { color: colors.danger }]}
      >
        {title}
      </Text>
    </Pressable>
  );
}

export function WarningBanner({ messages }: { messages: string[] }) {
  if (messages.length === 0) return null;
  return (
    <View style={styles.warning} accessibilityRole="alert">
      {messages.map((m) => (
        <Text key={m} style={styles.warningText}>
          {'⚠︎ '}
          {m}
        </Text>
      ))}
    </View>
  );
}

export function Row({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={styles.rowValue}>{value}</Text>
    </View>
  );
}

export const text = StyleSheet.create({
  title: { fontSize: 22, fontWeight: '700', color: colors.text },
  heading: { fontSize: 17, fontWeight: '600', color: colors.text, marginBottom: spacing.sm },
  body: { fontSize: 15, color: colors.text, lineHeight: 21 },
  muted: { fontSize: 13, color: colors.muted, lineHeight: 18 },
});

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  button: {
    backgroundColor: colors.primary,
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: spacing.md,
    alignItems: 'center',
    marginVertical: spacing.xs,
  },
  buttonSecondary: { backgroundColor: 'transparent', borderWidth: 1, borderColor: colors.primary },
  buttonDanger: { backgroundColor: 'transparent', borderWidth: 1, borderColor: colors.danger },
  buttonInverse: { backgroundColor: colors.surface },
  buttonText: { color: colors.primaryText, fontSize: 16, fontWeight: '600' },
  buttonTextAlt: { color: colors.primary },
  warning: {
    backgroundColor: colors.warningBg,
    borderRadius: 10,
    padding: spacing.md,
    marginBottom: spacing.md,
    gap: spacing.xs,
  },
  warningText: { color: colors.warningText, fontSize: 14, lineHeight: 19 },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 6,
    gap: spacing.md,
  },
  rowLabel: { color: colors.muted, fontSize: 14, flexShrink: 1 },
  rowValue: { color: colors.text, fontSize: 14, fontWeight: '600', textAlign: 'right' },
});
