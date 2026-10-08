import { Image, StyleSheet, View } from 'react-native';
import Svg, { Path } from 'react-native-svg';

import { colors } from '../theme';

// Proportions measured from the VisionGlyco logo: eye height 60/102 of its width,
// centre hole radius 17/102.
const ASPECT = 60 / 102;
const HOLE = 17 / 102;

/** The VisionGlyco eye mark, drawn as a vector so it stays sharp and can be tinted. */
export function EyeMark({ width, color = colors.brand }: { width: number; color?: string }) {
  const w = width;
  const h = w * ASPECT;
  const s = h / 2;
  const r = (s * s + (w / 2) ** 2) / (2 * s); // arc radius through the tips and top/bottom
  const cx = w / 2;
  const hole = HOLE * w;
  const d =
    `M0 ${s} A${r} ${r} 0 0 1 ${w} ${s} A${r} ${r} 0 0 1 0 ${s} Z ` +
    `M${cx - hole} ${s} a${hole} ${hole} 0 1 0 ${2 * hole} 0 a${hole} ${hole} 0 1 0 ${-2 * hole} 0 Z`;
  return (
    <Svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} accessibilityLabel="VisionGlyco">
      <Path d={d} fill={color} fillRule="evenodd" />
    </Svg>
  );
}

const WORDMARK_ASPECT = 1328 / 162;
const STACKED_ASPECT = 892 / 527;

/** Eye + VISIONGLYCO side by side, for headers. */
export function LogoInline({ height = 22 }: { height?: number }) {
  return (
    <View style={styles.inline} accessibilityRole="header" accessibilityLabel="VisionGlyco">
      <EyeMark width={height * 1.55} />
      <Image
        source={require('../../assets/brand/wordmark.png')}
        style={{ height: height * 0.62, width: height * 0.62 * WORDMARK_ASPECT }}
        resizeMode="contain"
      />
    </View>
  );
}

/** The full stacked logo, for the welcome screen. */
export function LogoStacked({ width = 180 }: { width?: number }) {
  return (
    <Image
      source={require('../../assets/brand/logo-stacked.png')}
      style={{ width, height: width / STACKED_ASPECT }}
      resizeMode="contain"
      accessibilityLabel="VisionGlyco"
    />
  );
}

const styles = StyleSheet.create({
  inline: { flexDirection: 'row', alignItems: 'center', gap: 10 },
});
