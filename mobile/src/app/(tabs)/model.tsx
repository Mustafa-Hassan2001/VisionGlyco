import { ScrollView, StyleSheet, Text } from 'react-native';

import { formatCi, formatDate, formatPercent } from '../../components/format';
import { Card, Row, WarningBanner, text } from '../../components/ui';
import { modelWarnings } from '../../ml/interpret';
import { getMetadata } from '../../ml/model';
import { spacing } from '../../theme';

export default function ModelScreen() {
  const metadata = getMetadata();

  if (!metadata) {
    return (
      <ScrollView contentContainerStyle={styles.container}>
        <Card>
          <Text style={text.heading}>No model installed</Text>
          <Text style={text.body}>
            This build has no trained model. Accuracy figures appear here once a model trained on
            labelled eye images is installed with ml/train.py --install-to-app.
          </Text>
        </Card>
      </ScrollView>
    );
  }

  const { dataset } = metadata;
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <WarningBanner messages={modelWarnings(metadata)} />

      <Card>
        <Text style={text.heading}>Held-out test results</Text>
        <Text style={[text.muted, { marginBottom: spacing.sm }]}>
          {metadata.evaluation ??
            `Measured on ${dataset.test.patients} patients the model never saw during training.`}
        </Text>
        {metadata.task === 'classify' ? (
          <>
            <Row
              label="Accuracy"
              value={formatPercent(metadata.test_metrics.accuracy) + formatCi(metadata.test_metrics.accuracy_ci95)}
            />
            <Row label="Always-guess-majority baseline"
              value={formatPercent(metadata.test_metrics.majority_baseline_accuracy)} />
            <Row label="AUC" value={metadata.test_metrics.auc?.toFixed(3) ?? '—'} />
            <Row label="Sensitivity" value={formatPercent(metadata.test_metrics.sensitivity)} />
            <Row label="Specificity" value={formatPercent(metadata.test_metrics.specificity)} />
            <Row label="Target accuracy" value={formatPercent(metadata.target_accuracy, 0)} />
          </>
        ) : (
          <>
            <Row label="Mean absolute error"
              value={`${metadata.test_metrics.mae.toFixed(1)} ${metadata.output.unit}`} />
            <Row label="Predict-the-mean baseline"
              value={`${metadata.test_metrics.mean_baseline_mae.toFixed(1)} ${metadata.output.unit}`} />
            <Row label="MARD" value={`${metadata.test_metrics.mard_percent.toFixed(1)}%`} />
            <Row label="Within ISO 15197 zone"
              value={`${metadata.test_metrics.within_iso15197_percent.toFixed(1)}% (needs 95%)`} />
          </>
        )}
      </Card>

      <Card>
        <Text style={text.heading}>What it predicts</Text>
        <Text style={text.body}>
          {metadata.task === 'classify'
            ? `Sorts each scan into "${metadata.output.positive_label}" or "${metadata.output.negative_label}".`
            : `An estimate of ${metadata.output.target_column} in ${metadata.output.unit}.`}
        </Text>
      </Card>

      {metadata.source && (
        <Card>
          <Text style={text.heading}>Data source</Text>
          <Text style={text.body}>{metadata.source.name}</Text>
          <Text style={[text.muted, { marginTop: spacing.xs }]}>{metadata.source.citation}</Text>
          <Text style={[text.muted, { marginTop: spacing.xs }]}>{metadata.source.url}</Text>
        </Card>
      )}

      <Card>
        <Text style={text.heading}>Training data</Text>
        {metadata.evaluation ? (
          <Row label="People (cross-validated)"
            value={`${dataset.train.images} images · ${dataset.train.patients} people`} />
        ) : (
          <>
            <Row label="Train" value={`${dataset.train.images} images · ${dataset.train.patients} patients`} />
            <Row label="Validation" value={`${dataset.val.images} images · ${dataset.val.patients} patients`} />
            <Row label="Test" value={`${dataset.test.images} images · ${dataset.test.patients} patients`} />
          </>
        )}
        <Row label="Backbone" value={metadata.backbone} />
        <Row label="Trained" value={formatDate(metadata.trained_at)} />
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: spacing.md },
});
