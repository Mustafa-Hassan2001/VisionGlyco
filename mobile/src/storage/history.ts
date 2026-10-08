import AsyncStorage from '@react-native-async-storage/async-storage';
import { Directory, File, Paths } from 'expo-file-system';

import type { ScanRecord } from '../ml/types';

// Scans never leave the device: records live in AsyncStorage, images in the
// app's document directory.
const KEY = 'visionglyco.scans.v1';

function scansDir(): Directory {
  const dir = new Directory(Paths.document, 'scans');
  dir.create({ intermediates: true, idempotent: true });
  return dir;
}

export async function listScans(): Promise<ScanRecord[]> {
  const raw = await AsyncStorage.getItem(KEY);
  const scans: ScanRecord[] = raw ? JSON.parse(raw) : [];
  return scans.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function getScan(id: string): Promise<ScanRecord | undefined> {
  return (await listScans()).find((s) => s.id === id);
}

export async function saveScan(
  scan: Omit<ScanRecord, 'id' | 'createdAt' | 'imageUri'>,
  tempImageUri: string,
): Promise<ScanRecord> {
  const createdAt = new Date().toISOString();
  const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const dest = new File(scansDir(), `${id}.jpg`);
  await new File(tempImageUri).copy(dest);
  const record: ScanRecord = { ...scan, id, createdAt, imageUri: dest.uri };
  const scans = await listScans();
  await AsyncStorage.setItem(KEY, JSON.stringify([record, ...scans]));
  return record;
}

export async function deleteScan(id: string): Promise<void> {
  const scans = await listScans();
  const target = scans.find((s) => s.id === id);
  if (target) {
    const file = new File(target.imageUri);
    if (file.exists) file.delete();
  }
  await AsyncStorage.setItem(KEY, JSON.stringify(scans.filter((s) => s.id !== id)));
}

export async function deleteAllScans(): Promise<void> {
  const dir = new Directory(Paths.document, 'scans');
  if (dir.exists) dir.delete();
  await AsyncStorage.removeItem(KEY);
}
