import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Alert, FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';

import { TrendChart } from '../../components/TrendChart';
import {
  formatDate,
  formatPercent,
  predictionDetail,
  predictionHeadline,
  predictionValue,
} from '../../components/format';
import { Button, Card, text } from '../../components/ui';
import { getMetadata } from '../../ml/model';
import type { ScanRecord } from '../../ml/types';
import { deleteAllScans, listScans } from '../../storage/history';
import { colors, spacing } from '../../theme';

export default function HistoryScreen() {
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const metadata = getMetadata();

  const reload = useCallback(() => {
    listScans().then(setScans).catch(() => setScans([]));
  }, []);
  useFocusEffect(reload);

  // Only chart scans from the currently installed model: scores from different
  // models are not comparable.
  const charted = scans.filter((s) => metadata && s.modelTrainedAt === metadata.trained_at);
  const isProbability = metadata?.task === 'classify';

  const clearAll = () =>
    Alert.alert('Delete all scans?', 'Photos and results will be removed from this device.', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => deleteAllScans().then(reload) },
    ]);

  return (
    <FlatList
      contentContainerStyle={styles.container}
      data={scans}
      keyExtractor={(s) => s.id}
      ListHeaderComponent={
        <Card>
          <Text style={text.heading}>Trend</Text>
          <TrendChart
            points={charted.map((s) => ({
              time: new Date(s.createdAt).getTime(),
              value: predictionValue(s.prediction),
            }))}
            min={isProbability ? 0 : undefined}
            max={isProbability ? 1 : undefined}
            threshold={metadata?.task === 'classify' ? metadata.output.threshold : undefined}
            formatValue={(v) => (isProbability ? formatPercent(v, 0) : v.toFixed(0))}
          />
          {isProbability && (
            <Text style={text.muted}>Model score per scan. Dashed line: decision threshold.</Text>
          )}
        </Card>
      }
      ListEmptyComponent={<Text style={styles.empty}>No scans yet.</Text>}
      renderItem={({ item }) => (
        <Pressable onPress={() => router.push(`/result/${item.id}`)}>
          <View style={styles.item}>
            <Image source={{ uri: item.imageUri }} style={styles.thumb} />
            <View style={{ flex: 1 }}>
              <Text style={styles.itemTitle}>{predictionHeadline(item.prediction)}</Text>
              <Text style={text.muted}>
                {formatDate(item.createdAt)} · {predictionDetail(item.prediction)}
              </Text>
            </View>
          </View>
        </Pressable>
      )}
      ListFooterComponent={
        scans.length > 0 ? <Button title="Delete all scans" variant="danger" onPress={clearAll} /> : null
      }
    />
  );
}

const styles = StyleSheet.create({
  container: { padding: spacing.md },
  empty: { color: colors.muted, textAlign: 'center', padding: spacing.lg },
  item: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
  },
  thumb: { width: 48, height: 48, borderRadius: 24, backgroundColor: colors.brandTint, borderWidth: 2, borderColor: colors.brand },
  itemTitle: { fontSize: 15, fontWeight: '600', color: colors.text },
});
