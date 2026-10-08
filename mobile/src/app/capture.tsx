import { CameraView, useCameraPermissions, type CameraType } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import { useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Button, text } from '../components/ui';
import { analyzeEyePhoto } from '../ml/analyzeEyePhoto';
import { getMetadata } from '../ml/model';
import { saveScan } from '../storage/history';
import { colors, spacing } from '../theme';

export default function CaptureScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const camera = useRef<CameraView>(null);
  const [facing, setFacing] = useState<CameraType>('back');
  const [torch, setTorch] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { width } = useWindowDimensions();
  const metadata = getMetadata();

  // The preview is 3:4, the same aspect as the photo, so the guide circle covers
  // exactly the centre crop the model sees.
  const previewW = width - spacing.md * 2;
  const previewH = (previewW * 4) / 3;
  const guide = previewW * (metadata?.input.crop_fraction ?? 0.8);

  const analyze = async (uri: string, croppedToIris = false) => {
    setBusy(true);
    setError(null);
    try {
      const { prediction, thumbnailUri } = await analyzeEyePhoto(uri, { croppedToIris });
      const record = await saveScan(
        { prediction, modelTrainedAt: metadata?.trained_at ?? '' },
        thumbnailUri,
      );
      router.replace(`/result/${record.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  const takePhoto = async () => {
    const photo = await camera.current?.takePictureAsync({ quality: 1 });
    if (photo) await analyze(photo.uri);
  };

  const pickPhoto = async () => {
    // The square crop editor lets the user frame the iris exactly.
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
      allowsEditing: true,
      aspect: [1, 1],
    });
    if (!result.canceled && result.assets[0]) await analyze(result.assets[0].uri, true);
  };

  if (!permission) {
    return <View style={styles.center}><ActivityIndicator color={colors.primary} /></View>;
  }

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      {permission.granted ? (
        <View style={{ width: previewW, height: previewH, alignSelf: 'center' }}>
          <CameraView
            ref={camera}
            style={StyleSheet.absoluteFill}
            facing={facing}
            enableTorch={torch}
            autofocus="on"
          />
          <View pointerEvents="none" style={styles.overlay}>
            <View style={[styles.guide, { width: guide, height: guide, borderRadius: guide / 2 }]} />
            <View style={styles.centreDot} />
          </View>
        </View>
      ) : (
        <View style={styles.permission}>
          <Text style={text.body}>Camera access is needed to photograph your eye.</Text>
          <Button title="Allow camera" onPress={requestPermission} />
        </View>
      )}

      <View style={styles.controls}>
        {busy ? (
          <View style={styles.busy}>
            <ActivityIndicator color={colors.primary} />
            <Text style={text.body}>Analysing on device…</Text>
          </View>
        ) : (
          <>
            {error && <Text style={styles.error}>{error}</Text>}
            <Text style={[text.muted, { textAlign: 'center' }]}>
              Line the coloured edge of your iris up with the circle and centre the pupil on the
              dot. Hold steady and avoid glare. For a gallery photo, crop the square tightly
              around the iris.
            </Text>
            {permission.granted && <Button title="Capture" onPress={takePhoto} />}
            <View style={styles.row}>
              {permission.granted && (
                <>
                  <View style={{ flex: 1 }}>
                    <Button title={torch ? 'Light off' : 'Light on'} variant="secondary"
                      onPress={() => setTorch((t) => !t)} />
                  </View>
                  <View style={{ flex: 1 }}>
                    <Button title="Flip" variant="secondary"
                      onPress={() => setFacing((f) => (f === 'back' ? 'front' : 'back'))} />
                  </View>
                </>
              )}
              <View style={{ flex: 1 }}>
                <Button title="Gallery" variant="secondary" onPress={pickPhoto} />
              </View>
            </View>
            <Button title="Cancel" variant="secondary" onPress={() => router.back()} />
          </>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background, paddingTop: spacing.md },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  overlay: { ...StyleSheet.absoluteFill, alignItems: 'center', justifyContent: 'center' },
  guide: { borderWidth: 3, borderColor: colors.brand },
  centreDot: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.brand,
  },
  permission: { padding: spacing.lg, gap: spacing.md },
  controls: { padding: spacing.md, gap: spacing.xs },
  row: { flexDirection: 'row', gap: spacing.sm },
  busy: { alignItems: 'center', gap: spacing.sm, padding: spacing.lg },
  error: { color: colors.danger, textAlign: 'center', marginBottom: spacing.sm },
});
