import { Tabs } from 'expo-router/js-tabs';
import type { ColorValue } from 'react-native';
import Svg, { Circle, Path, Rect } from 'react-native-svg';

import { EyeMark, LogoInline } from '../../components/Brand';
import { colors } from '../../theme';

type IconProps = { color: ColorValue };

function HistoryIcon({ color }: IconProps) {
  return (
    <Svg width={24} height={24} viewBox="0 0 24 24" fill="none" stroke={color as string} strokeWidth={2}>
      <Circle cx={12} cy={12} r={9} />
      <Path d="M12 7v5l3 2" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

function ModelIcon({ color }: IconProps) {
  return (
    <Svg width={24} height={24} viewBox="0 0 24 24" fill="none" stroke={color as string} strokeWidth={2}>
      <Rect x={4} y={13} width={4} height={7} rx={1} />
      <Rect x={10} y={8} width={4} height={12} rx={1} />
      <Rect x={16} y={4} width={4} height={16} rx={1} />
    </Svg>
  );
}

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: colors.primary,
        tabBarInactiveTintColor: colors.muted,
        tabBarStyle: { backgroundColor: colors.surface, borderTopColor: colors.border },
        headerStyle: { backgroundColor: colors.surface },
        headerTintColor: colors.text,
        headerTitleStyle: { color: colors.text, fontWeight: '700' },
        sceneStyle: { backgroundColor: colors.background },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'VisionGlyco',
          headerTitle: () => <LogoInline height={24} />,
          tabBarLabel: 'Home',
          tabBarIcon: ({ color }) => <EyeMark width={26} color={color as string} />,
        }}
      />
      <Tabs.Screen
        name="history"
        options={{ title: 'History', tabBarIcon: ({ color }) => <HistoryIcon color={color} /> }}
      />
      <Tabs.Screen
        name="model"
        options={{ title: 'Model', tabBarIcon: ({ color }) => <ModelIcon color={color} /> }}
      />
    </Tabs>
  );
}
