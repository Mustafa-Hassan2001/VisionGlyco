import AsyncStorage from '@react-native-async-storage/async-storage';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

import { LogoStacked } from '../components/Brand';
import { Button, text } from '../components/ui';
import { colors, spacing } from '../theme';

const DISCLAIMER_KEY = 'visionglyco.disclaimerAccepted.v1';

export default function RootLayout() {
  const [accepted, setAccepted] = useState<boolean | null>(null);

  useEffect(() => {
    AsyncStorage.getItem(DISCLAIMER_KEY)
      .then((v) => setAccepted(v === 'yes'))
      .catch(() => setAccepted(false));
  }, []);

  const accept = () => {
    setAccepted(true);
    AsyncStorage.setItem(DISCLAIMER_KEY, 'yes').catch(() => {});
  };

  return (
    <SafeAreaProvider>
      <StatusBar style="dark" />
      {accepted === null ? (
        <View style={styles.center}>
          <ActivityIndicator color={colors.primary} />
        </View>
      ) : accepted ? (
        <Stack
          screenOptions={{
            headerTintColor: colors.primary,
            headerStyle: { backgroundColor: colors.surface },
            headerTitleStyle: { color: colors.text, fontWeight: '700' },
            contentStyle: { backgroundColor: colors.background },
          }}
        >
          <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
          <Stack.Screen name="capture" options={{ title: 'Scan eye', presentation: 'fullScreenModal' }} />
          <Stack.Screen name="result/[id]" options={{ title: 'Result' }} />
        </Stack>
      ) : (
        <Disclaimer onAccept={accept} />
      )}
    </SafeAreaProvider>
  );
}

function Disclaimer({ onAccept }: { onAccept: () => void }) {
  return (
    <SafeAreaView style={styles.disclaimer}>
      <ScrollView contentContainerStyle={{ padding: spacing.lg }}>
        <View style={styles.logo}>
          <LogoStacked width={200} />
        </View>
        <Text style={text.title}>Welcome</Text>
        <Text style={[text.body, styles.p]}>
          VisionGlyco analyses a photo of your eye with an AI model to screen for diabetes-related
          patterns.
        </Text>
        <Text style={[text.body, styles.p]}>
          VisionGlyco is a research prototype, not a medical device. Always confirm results with a
          blood glucose test and follow your doctor’s advice.
        </Text>
        <Text style={[text.muted, styles.p]}>
          Photos and results are stored only on this device.
        </Text>
        <Button title="I understand" onPress={onAccept} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.surface },
  disclaimer: { flex: 1, backgroundColor: colors.surface },
  p: { marginVertical: spacing.sm },
  logo: { alignItems: 'center', marginTop: spacing.lg, marginBottom: spacing.lg },
});
