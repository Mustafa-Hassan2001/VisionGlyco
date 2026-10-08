import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { formatDate, predictionDetail, predictionHeadline } from '../../components/format';
import { EyeMark } from '../../components/Brand';
import { Button, Card, WarningBanner, text } from '../../components/ui';
import { modelWarnings } from '../../ml/interpret';
import { getMetadata, isModelInstalled } from '../../ml/model';
import type { ScanRecord } from '../../ml/types';
import { listScans } from '../../storage/history';
import { colors, spacing } from '../../theme';

export default function HomeScreen() {
  const [latest, setLatest] = useState<ScanRecord | null>(null);
  const metadata = getMetadata();
  const installed = isModelInstalled();

  useFocusEffect(
    useCallback(() => {
      listScans().then((s) => setLatest(s[0] ?? null)).catch(() => setLatest(null));
    }, []),
  );

  return (
    <ScrollView contentContainerStyle={styles.container}>
      {metadata && <WarningBanner messages={modelWarnings(metadata)} />}

      <Card style={styles.scanCard}>
        <EyeMark width={56} color={colors.primaryText} />
        <Text style={[text.heading, styles.onBlue, { marginTop: spacing.sm }]}>Scan your eye</Text>
        <Text style={[text.body, styles.onBlueMuted, { marginBottom: spacing.md }]}>
          Attach the iris imaging adapter, use good even lighting, and line your iris up with the
          guide circle, or upload a photo and crop it to the iris. The photo is analysed on this
          phone; nothing is uploaded.
        </Text>
        {installed ? (
          <Button title="Start scan" variant="inverse" onPress={() => router.push('/capture')} />
        ) : (
          <View style={styles.noModel}>
            <Text style={[text.body, styles.onBlue]}>No AI model is installed in this build.</Text>
            <Text style={[text.muted, styles.onBlueMuted]}>
              Train one on labelled eye images with ml/train.py, then run it with
              --install-to-app ../mobile and rebuild the app. See ml/README.md.
            </Text>
          </View>
        )}
      </Card>

      {latest && (
        <Pressable onPress={() => router.push(`/result/${latest.id}`)}>
          <Card style={styles.latest}>
            <Image source={{ uri: latest.imageUri }} style={styles.thumb} />
            <View style={{ flex: 1 }}>
              <Text style={text.muted}>Last scan · {formatDate(latest.createdAt)}</Text>
              <Text style={styles.headline}>{predictionHeadline(latest.prediction)}</Text>
              <Text style={text.muted}>{predictionDetail(latest.prediction)}</Text>
            </View>
          </Card>
        </Pressable>
      )}

      <Text style={[text.muted, styles.footer]}>
        Research prototype, not a medical device. Confirm any result with a blood glucose meter or
        an HbA1c lab test.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: spacing.md },
  scanCard: { backgroundColor: colors.primary, borderColor: colors.primary, padding: spacing.lg },
  onBlue: { color: colors.primaryText },
  onBlueMuted: { color: colors.onPrimaryMuted },
  noModel: { gap: spacing.sm, padding: spacing.md, borderRadius: 10, backgroundColor: colors.primaryPressed },
  latest: { flexDirection: 'row', gap: spacing.md, alignItems: 'center' },
  thumb: { width: 64, height: 64, borderRadius: 32, backgroundColor: colors.brandTint, borderWidth: 2, borderColor: colors.brand },
  headline: { fontSize: 17, fontWeight: '600', color: colors.text, marginVertical: 2 },
  footer: { textAlign: 'center', marginTop: spacing.sm },
});
