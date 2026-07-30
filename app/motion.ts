export type GeoFix = {
  latitude: number;
  longitude: number;
  accuracy: number;
  speedMps: number | null;
  timestamp: number;
};

export type MotionSnapshot = {
  speedMps: number;
  reliable: boolean;
  accuracy: number | null;
  updatedAt: number;
};

export const EMPTY_MOTION: MotionSnapshot = {
  speedMps: 0,
  reliable: false,
  accuracy: null,
  updatedAt: 0,
};

const EARTH_RADIUS_METERS = 6_371_000;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function toRadians(value: number) {
  return (value * Math.PI) / 180;
}

export function distanceBetweenFixes(from: GeoFix, to: GeoFix) {
  const latitudeDelta = toRadians(to.latitude - from.latitude);
  const longitudeDelta = toRadians(to.longitude - from.longitude);
  const fromLatitude = toRadians(from.latitude);
  const toLatitude = toRadians(to.latitude);
  const haversine =
    Math.sin(latitudeDelta / 2) ** 2 +
    Math.cos(fromLatitude) *
      Math.cos(toLatitude) *
      Math.sin(longitudeDelta / 2) ** 2;

  return (
    2 *
    EARTH_RADIUS_METERS *
    Math.asin(Math.min(1, Math.sqrt(haversine)))
  );
}

export function deriveMotion(
  previousFix: GeoFix | null,
  currentFix: GeoFix,
  previousMotion: MotionSnapshot,
): MotionSnapshot {
  const accurateEnough = currentFix.accuracy <= 55;
  let rawSpeed =
    currentFix.speedMps !== null &&
    Number.isFinite(currentFix.speedMps) &&
    currentFix.speedMps >= 0
      ? currentFix.speedMps
      : null;

  if (rawSpeed === null && previousFix) {
    const elapsedSeconds =
      (currentFix.timestamp - previousFix.timestamp) / 1000;
    const fixesAccurateEnough =
      Math.max(previousFix.accuracy, currentFix.accuracy) <= 55;

    if (
      fixesAccurateEnough &&
      elapsedSeconds >= 0.45 &&
      elapsedSeconds <= 12
    ) {
      const measuredDistance = distanceBetweenFixes(
        previousFix,
        currentFix,
      );
      const jitterAllowance =
        Math.max(previousFix.accuracy, currentFix.accuracy) * 0.28;
      rawSpeed =
        Math.max(0, measuredDistance - jitterAllowance) / elapsedSeconds;
    }
  }

  if (!accurateEnough || rawSpeed === null) {
    return {
      ...previousMotion,
      reliable: false,
      accuracy: currentFix.accuracy,
      updatedAt: currentFix.timestamp,
    };
  }

  rawSpeed = clamp(rawSpeed, 0, 70);
  if (rawSpeed < 0.65 && currentFix.accuracy > 12) rawSpeed = 0;

  const rising = rawSpeed > previousMotion.speedMps;
  const smoothing = rising ? 0.48 : 0.34;
  let speedMps =
    !previousMotion.reliable
      ? rawSpeed
      : previousMotion.speedMps * (1 - smoothing) + rawSpeed * smoothing;

  if (rawSpeed < 0.25 && speedMps < 0.8) speedMps = 0;

  return {
    speedMps,
    reliable: true,
    accuracy: currentFix.accuracy,
    updatedAt: currentFix.timestamp,
  };
}

export function freshMotion(
  motion: MotionSnapshot,
  now = Date.now(),
): MotionSnapshot {
  if (!motion.reliable || now - motion.updatedAt > 6000) {
    return { ...motion, reliable: false };
  }
  return motion;
}

export function formatSpeed(speedMps: number) {
  return Math.max(0, Math.round(speedMps * 3.6));
}
