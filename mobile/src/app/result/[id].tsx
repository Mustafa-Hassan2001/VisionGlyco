import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useState } from 'react';
import { Alert, Image, ScrollView, StyleSheet, Text } from 'react-native';

import { formatDate, predictionDetail, predictionHeadline } from '../../components/format';
import { Button, Card, Row, WarningBanner, text } from '../../components/ui';
import { modelWarnings } from '../../ml/interpret';
import { getMetadata } from '../../ml/model';
import type { ScanRecord } from '../../ml/types';
import { deleteScan, getScan } from '../../storage/history';
import { colors, spacing } from '../../theme';

export default function ResultScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [scan, setScan] = useState<ScanRecord | null | undefined>(undefined);
  const metadata = getMetadata();

  useEffect(() => {
    getScan(id).then((s) => setScan(s ?? null)).catch(() => setScan(null));
  }, [id]);

  if (scan === undefined) return null;
  if (scan === null) {
    return <Text style={[text.body, { padding: spacing.lg }]}>This scan no longer exists.</Text>;
  }

  const { prediction } = scan;
  const flagged = prediction.kind === 'probability' && prediction.positive;
  const sameModel = metadata?.trained_at === scan.modelTrainedAt;

  const remove = () =>
    Alert.alert('Delete this scan?', undefined, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => deleteScan(scan.id).then(() => router.back()) },
    ]);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      {metadata && <WarningBanner messages={modelWarnings(metadata)} />}

      <Image source={{ uri: scan.imageUri }} style={styles.image} />

      <Card>
        <Text style={text.muted}>{formatDate(scan.createdAt)}</Text>
        <Text style={[styles.headline, { color: flagged ? colors.positive : colors.text }]}>
          {predictionHeadline(prediction)}
        </Text>
        <Text style={text.muted}>{predictionDetail(prediction)}</Text>
      </Card>

      <Card>
        <Text style={text.heading}>What to do next</Text>
        <Text style={text.body}>
          {flagged
            ? 'The model flagged this scan. This is not a diagnosis. Please arrange an HbA1c or fasting blood glucose test with your doctor.'
            : 'This is not an all-clear. Keep following your normal testing routine and your doctor’s advice.'}
        </Text>
      </Card>

      {metadata?.task === 'classify' && sameModel && (
        <Card>
          <Text style={text.heading}>How reliable is this?</Text>
          <Row label="Model accuracy on unseen patients"
            value={`${(metadata.test_metrics.accuracy * 100).toFixed(1)}%`} />
          <Row label="Patients in test set" value={String(metadata.dataset.test.patients)} />
        </Card>
      )}
      {!sameModel && (
        <Text style={[text.muted, { marginBottom: spacing.md }]}>
          This scan was analysed by an earlier model version.
        </Text>
      )}

      <Button title="Delete scan" variant="danger" onPress={remove} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: spacing.md },
  image: {
    width: 200,
    height: 200,
    borderRadius: 100,
    alignSelf: 'center',
    marginBottom: spacing.md,
    backgroundColor: colors.brandTint,
    borderWidth: 4,
    borderColor: colors.brand,
  },
  headline: { fontSize: 24, fontWeight: '700', marginVertical: spacing.xs },
});
