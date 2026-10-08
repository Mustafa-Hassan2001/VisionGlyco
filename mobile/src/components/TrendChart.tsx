import { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Svg, { Circle, Line, Polyline, Text as SvgText } from 'react-native-svg';

import { colors } from '../theme';

export type TrendPoint = { time: number; value: number };

const HEIGHT = 180;
const PAD = { top: 12, right: 12, bottom: 20, left: 40 };

export function TrendChart({
  points,
  min,
  max,
  threshold,
  formatValue,
}: {
  points: TrendPoint[];
  min?: number;
  max?: number;
  threshold?: number;
  formatValue: (v: number) => string;
}) {
  const [width, setWidth] = useState(0);
  if (points.length < 2) {
    return <Text style={styles.empty}>Scan at least twice to see a trend.</Text>;
  }

  const sorted = [...points].sort((a, b) => a.time - b.time);
  const values = sorted.map((p) => p.value);
  let lo = min ?? Math.min(...values);
  let hi = max ?? Math.max(...values);
  if (hi - lo < 1e-6) {
    lo -= 1;
    hi += 1;
  }
  const t0 = sorted[0].time;
  const t1 = sorted[sorted.length - 1].time;
  const innerW = Math.max(1, width - PAD.left - PAD.right);
  const innerH = HEIGHT - PAD.top - PAD.bottom;
  const x = (t: number) => PAD.left + (t1 === t0 ? innerW / 2 : ((t - t0) / (t1 - t0)) * innerW);
  const y = (v: number) => PAD.top + (1 - (v - lo) / (hi - lo)) * innerH;

  return (
    <View onLayout={(e) => setWidth(e.nativeEvent.layout.width)} accessibilityLabel="Trend chart">
      {width > 0 && (
        <Svg width={width} height={HEIGHT}>
          {[lo, (lo + hi) / 2, hi].map((v) => (
            <SvgText key={v} x={PAD.left - 6} y={y(v) + 4} fontSize={10} fill={colors.muted} textAnchor="end">
              {formatValue(v)}
            </SvgText>
          ))}
          <Line x1={PAD.left} y1={PAD.top + innerH} x2={width - PAD.right} y2={PAD.top + innerH}
            stroke={colors.border} />
          {threshold != null && threshold >= lo && threshold <= hi && (
            <Line x1={PAD.left} y1={y(threshold)} x2={width - PAD.right} y2={y(threshold)}
              stroke={colors.danger} strokeDasharray="4 4" />
          )}
          <Polyline
            points={sorted.map((p) => `${x(p.time)},${y(p.value)}`).join(' ')}
            fill="none"
            stroke={colors.brand}
            strokeWidth={2}
          />
          {sorted.map((p) => (
            <Circle key={p.time} cx={x(p.time)} cy={y(p.value)} r={3.5} fill={colors.brand} />
          ))}
          <SvgText x={PAD.left} y={HEIGHT - 4} fontSize={10} fill={colors.muted}>
            {new Date(t0).toLocaleDateString()}
          </SvgText>
          <SvgText x={width - PAD.right} y={HEIGHT - 4} fontSize={10} fill={colors.muted} textAnchor="end">
            {new Date(t1).toLocaleDateString()}
          </SvgText>
        </Svg>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  empty: { color: colors.muted, fontSize: 14, paddingVertical: 24, textAlign: 'center' },
});
