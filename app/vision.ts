import { formatSpeed, type MotionSnapshot } from "./motion";
import {
  dashboardTopAt,
  laneBoundsAt,
  type DashboardMask,
  type DetectedLane,
  type NormalizedPoint,
  type RoadScene,
} from "./road-scene";

export type TravelMode = "walk" | "ride" | "drive";
export type RiskLevel = "info" | "safe" | "watch" | "danger";

export type Box = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
  cls?: number;
  name: string;
};

export type TrackRecord = {
  id: number;
  name: string;
  cls?: number;
  cx: number;
  cy: number;
  bottom: number;
  area: number;
  scale: number;
  vx: number;
  vy: number;
  expansionRate: number;
  distance: number | null;
  seenCount: number;
  watchEvidence: number;
  dangerEvidence: number;
  stableRisk: RiskLevel;
  updatedAt: number;
  history: Array<{ x: number; y: number }>;
};

export type TrackStore = {
  nextId: number;
  tracks: Map<number, TrackRecord>;
};

export type AnalyzedDetection = Box & {
  id: number;
  thaiName: string;
  distance: number;
  ttc: number;
  risk: RiskLevel;
  inPath: boolean;
  predictedInPath: boolean;
  direction: "ซ้าย" | "กลาง" | "ขวา";
  severity: number;
  history: Array<{ x: number; y: number }>;
};

export type VisionAlert = {
  key: string;
  level: "watch" | "danger";
  title: string;
  detail: string;
  objectName: string;
  direction: string;
  distance: number;
};

const MODE_PROFILE = {
  walk: {
    corridorWidth: 0.42,
    cameraHeight: 1.55,
    warningDistance: 5.5,
    dangerDistance: 2.2,
    warningTtc: 3.2,
    dangerTtc: 1.55,
    predictionSeconds: 1,
  },
  ride: {
    corridorWidth: 0.35,
    cameraHeight: 1.25,
    warningDistance: 14,
    dangerDistance: 5.5,
    warningTtc: 4,
    dangerTtc: 1.9,
    predictionSeconds: 1.25,
  },
  drive: {
    corridorWidth: 0.3,
    cameraHeight: 1.2,
    warningDistance: 24,
    dangerDistance: 9,
    warningTtc: 4.5,
    dangerTtc: 2.15,
    predictionSeconds: 1.5,
  },
} as const;

const THAI_NAMES: Record<string, string> = {
  person: "คน",
  bicycle: "จักรยาน",
  car: "รถยนต์",
  motorcycle: "มอเตอร์ไซค์",
  airplane: "เครื่องบิน",
  bus: "รถบัส",
  train: "รถไฟ",
  truck: "รถบรรทุก",
  boat: "เรือ",
  "traffic light": "สัญญาณไฟ",
  "fire hydrant": "หัวจ่ายน้ำ",
  "stop sign": "ป้ายหยุด",
  "parking meter": "มิเตอร์จอดรถ",
  bench: "ม้านั่ง",
  bird: "นก",
  cat: "แมว",
  dog: "สุนัข",
  horse: "ม้า",
  sheep: "แกะ",
  cow: "วัว",
  elephant: "ช้าง",
  bear: "หมี",
  zebra: "ม้าลาย",
  giraffe: "ยีราฟ",
  backpack: "กระเป๋าเป้",
  umbrella: "ร่ม",
  handbag: "กระเป๋า",
  suitcase: "กระเป๋าเดินทาง",
  "sports ball": "ลูกบอล",
  skateboard: "สเกตบอร์ด",
  chair: "เก้าอี้",
  couch: "โซฟา",
  "potted plant": "กระถางต้นไม้",
  bed: "เตียง",
  "dining table": "โต๊ะ",
};

const PHYSICAL_HEIGHT: Record<string, number> = {
  person: 1.7,
  bicycle: 1.15,
  car: 1.5,
  motorcycle: 1.35,
  bus: 3.1,
  train: 3.4,
  truck: 3.2,
  bench: 0.85,
  bird: 0.3,
  cat: 0.35,
  dog: 0.65,
  horse: 1.7,
  sheep: 0.9,
  cow: 1.5,
  elephant: 2.8,
  bear: 1.5,
  zebra: 1.45,
  giraffe: 4.2,
  backpack: 0.55,
  umbrella: 1,
  handbag: 0.45,
  suitcase: 0.7,
  "sports ball": 0.22,
  skateboard: 0.15,
  chair: 0.9,
  couch: 0.9,
  "potted plant": 0.75,
  bed: 0.65,
  "dining table": 0.75,
};

const RISK_COLOR: Record<RiskLevel, string> = {
  info: "#72e6ff",
  safe: "#ffe071",
  watch: "#ffab4a",
  danger: "#ff4f64",
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function thaiName(name: string) {
  if (name.toLowerCase() === "obstacle") return "สิ่งกีดขวาง";
  return THAI_NAMES[name.toLowerCase()] ?? "สิ่งกีดขวาง";
}

function classHeight(name: string) {
  return PHYSICAL_HEIGHT[name.toLowerCase()] ?? 1.25;
}

function directionFor(cx: number, width: number) {
  if (cx < width * 0.41) return "ซ้าย" as const;
  if (cx > width * 0.59) return "ขวา" as const;
  return "กลาง" as const;
}

function riskThresholds(
  mode: TravelMode,
  motion: MotionSnapshot,
) {
  const profile = MODE_PROFILE[mode];
  if (!motion.reliable) {
    return {
      warningDistance: profile.warningDistance,
      dangerDistance: profile.dangerDistance,
      moving: true,
    };
  }

  const speed = clamp(motion.speedMps, 0, 55);
  const settings = {
    walk: {
      movementFloor: 0.35,
      reactionSeconds: 0.75,
      deceleration: 2.3,
      warningMargin: 2.6,
      dangerMargin: 1.2,
      warningMinimum: 4,
      dangerMinimum: 1.7,
      warningMaximum: 9,
      dangerMaximum: 4.5,
    },
    ride: {
      movementFloor: 1.1,
      reactionSeconds: 1,
      deceleration: 3.2,
      warningMargin: 4,
      dangerMargin: 2.4,
      warningMinimum: 7,
      dangerMinimum: 3.2,
      warningMaximum: 34,
      dangerMaximum: 17,
    },
    drive: {
      movementFloor: 1.7,
      reactionSeconds: 1.2,
      deceleration: 4.5,
      warningMargin: 5,
      dangerMargin: 3.2,
      warningMinimum: 9,
      dangerMinimum: 4.5,
      warningMaximum: 65,
      dangerMaximum: 32,
    },
  }[mode];
  const stoppingDistance =
    speed * settings.reactionSeconds +
    (speed * speed) / (2 * settings.deceleration);

  return {
    warningDistance: clamp(
      stoppingDistance * 1.08 + settings.warningMargin,
      settings.warningMinimum,
      settings.warningMaximum,
    ),
    dangerDistance: clamp(
      stoppingDistance * 0.54 + settings.dangerMargin,
      settings.dangerMinimum,
      settings.dangerMaximum,
    ),
    moving: speed >= settings.movementFloor,
  };
}

export function corridorAt(
  y: number,
  width: number,
  height: number,
  mode: TravelMode,
  roadScene?: RoadScene | null,
) {
  const detectedLane = mode === "drive" ? roadScene?.lane : null;
  if (detectedLane) {
    const normalizedY = clamp(y / height, 0, 1);
    const bounds = laneBoundsAt(detectedLane, normalizedY);
    const margin = width * 0.018;
    const left = bounds.left * width - margin;
    const right = bounds.right * width + margin;
    return {
      left,
      right,
      center: (left + right) / 2,
      half: (right - left) / 2,
      horizon: detectedLane.topY * height,
      bottom: detectedLane.bottomY * height,
    };
  }

  const profile = MODE_PROFILE[mode];
  const horizon = height * 0.42;
  const bottom = height * 0.985;
  const center = width * 0.5;
  const farHalf = width * 0.055;
  const nearHalf = width * profile.corridorWidth;
  const progress = clamp((y - horizon) / Math.max(1, bottom - horizon), 0, 1);
  const eased = Math.pow(progress, 1.08);
  const half = farHalf + (nearHalf - farHalf) * eased;
  return {
    left: center - half,
    right: center + half,
    center,
    half,
    horizon,
    bottom,
  };
}

function isDashboardDetection(
  box: Box,
  width: number,
  height: number,
  mode: TravelMode,
  roadScene?: RoadScene | null,
) {
  const dashboard =
    mode === "drive" ? roadScene?.dashboard : null;
  if (!dashboard) return false;
  const centerX = clamp((box.x1 + box.x2) / 2 / width, 0, 1);
  const dashboardY = dashboardTopAt(dashboard, centerX) * height;
  const boxHeight = Math.max(1, box.y2 - box.y1);
  const overlap = Math.max(0, box.y2 - dashboardY) / boxHeight;
  const centerY = (box.y1 + box.y2) / 2;
  return (
    centerY >= dashboardY + height * 0.006 ||
    (overlap >= 0.58 && box.y1 >= dashboardY - height * 0.12)
  );
}

function estimateDistance(
  box: Box,
  width: number,
  height: number,
  mode: TravelMode,
) {
  const profile = MODE_PROFILE[mode];
  const boxHeight = Math.max(2, box.y2 - box.y1);
  const apparentDistance =
    (classHeight(box.name) * height * 0.88) / boxHeight;
  const horizon = height * 0.42;
  const groundOffset = box.y2 - horizon;

  if (groundOffset <= height * 0.025) {
    return clamp(apparentDistance, 0.6, 120);
  }

  const groundDistance =
    (profile.cameraHeight * height * 1.65) /
    Math.max(height * 0.035, groundOffset);
  const apparentWeight =
    box.name === "person" || box.name === "car" || box.name === "truck"
      ? 0.38
      : 0.25;

  return clamp(
    groundDistance * (1 - apparentWeight) +
      apparentDistance * apparentWeight,
    0.6,
    120,
  );
}

function overlapRatio(
  leftA: number,
  rightA: number,
  leftB: number,
  rightB: number,
) {
  const overlap = Math.max(0, Math.min(rightA, rightB) - Math.max(leftA, leftB));
  return overlap / Math.max(1, rightA - leftA);
}

function matchTrack(
  box: Box,
  now: number,
  store: TrackStore,
  usedTracks: Set<number>,
  frameDiagonal: number,
) {
  const cx = (box.x1 + box.x2) / 2;
  const cy = (box.y1 + box.y2) / 2;
  const area = Math.max(1, (box.x2 - box.x1) * (box.y2 - box.y1));
  let best: TrackRecord | undefined;
  let bestScore = Number.POSITIVE_INFINITY;

  for (const track of store.tracks.values()) {
    if (
      usedTracks.has(track.id) ||
      track.name !== box.name ||
      track.cls !== box.cls ||
      now - track.updatedAt > 1500
    ) {
      continue;
    }
    const centerDistance = Math.hypot(cx - track.cx, cy - track.cy);
    const areaRatio = area / Math.max(1, track.area);
    const score =
      centerDistance / frameDiagonal + Math.abs(Math.log(areaRatio)) * 0.14;
    if (score < 0.28 && score < bestScore) {
      best = track;
      bestScore = score;
    }
  }
  return best;
}

function updateTrack(
  box: Box,
  now: number,
  previous: TrackRecord | undefined,
  store: TrackStore,
) {
  const cx = (box.x1 + box.x2) / 2;
  const cy = (box.y1 + box.y2) / 2;
  const bottom = box.y2;
  const area = Math.max(1, (box.x2 - box.x1) * (box.y2 - box.y1));
  const scale = Math.sqrt(area);

  if (!previous) {
    const created: TrackRecord = {
      id: store.nextId++,
      name: box.name,
      cls: box.cls,
      cx,
      cy,
      bottom,
      area,
      scale,
      vx: 0,
      vy: 0,
      expansionRate: 0,
      distance: null,
      seenCount: 1,
      watchEvidence: 0,
      dangerEvidence: 0,
      stableRisk: "safe",
      updatedAt: now,
      history: [{ x: cx, y: bottom }],
    };
    store.tracks.set(created.id, created);
    return created;
  }

  const dt = clamp((now - previous.updatedAt) / 1000, 0.03, 1.5);
  const rawVx = (cx - previous.cx) / dt;
  const rawVy = (bottom - previous.bottom) / dt;
  const rawExpansion = clamp(
    (scale - previous.scale) / Math.max(1, previous.scale * dt),
    -1.2,
    1.2,
  );
  const history = [
    ...previous.history.slice(-5),
    { x: cx, y: bottom },
  ];
  const updated: TrackRecord = {
    ...previous,
    cx,
    cy,
    bottom,
    area,
    scale,
    vx: previous.vx * 0.58 + rawVx * 0.42,
    vy: previous.vy * 0.58 + rawVy * 0.42,
    expansionRate:
      previous.expansionRate * 0.7 + rawExpansion * 0.3,
    seenCount: previous.seenCount + 1,
    updatedAt: now,
    history,
  };
  store.tracks.set(updated.id, updated);
  return updated;
}

function assessRisk(
  box: Box,
  track: TrackRecord,
  distance: number,
  width: number,
  height: number,
  mode: TravelMode,
  motion: MotionSnapshot,
  roadScene?: RoadScene | null,
) {
  const profile = MODE_PROFILE[mode];
  const thresholds = riskThresholds(mode, motion);
  const boxWidth = Math.max(1, box.x2 - box.x1);
  const cx = (box.x1 + box.x2) / 2;
  const currentCorridor = corridorAt(
    box.y2,
    width,
    height,
    mode,
    roadScene,
  );
  const currentOverlap = overlapRatio(
    box.x1,
    box.x2,
    currentCorridor.left,
    currentCorridor.right,
  );
  const inPath =
    box.y2 > currentCorridor.horizon &&
    (currentOverlap > 0.22 ||
      (cx > currentCorridor.left && cx < currentCorridor.right));

  const lookAhead = profile.predictionSeconds;
  const predictedX = cx + track.vx * lookAhead;
  const predictedBottom = clamp(
    box.y2 + track.vy * lookAhead,
    0,
    height,
  );
  const predictedCorridor = corridorAt(
    predictedBottom,
    width,
    height,
    mode,
    roadScene,
  );
  const predictedInPath =
    predictedBottom > predictedCorridor.horizon &&
    predictedX + boxWidth * 0.3 > predictedCorridor.left &&
    predictedX - boxWidth * 0.3 < predictedCorridor.right;

  const ttc =
    track.seenCount >= 3 && track.expansionRate > 0.05
      ? clamp(1 / track.expansionRate, 0.1, 99)
      : 99;
  // The fast obstacle model intentionally does not identify object classes.
  // Every sufficiently confident box is treated as a possible obstacle.
  const isHazard = true;
  const isInformational = false;
  const priorityScale = 1;
  const warningDistance = thresholds.warningDistance * priorityScale;
  const dangerDistance = thresholds.dangerDistance * priorityScale;
  const nearBottom =
    box.y2 >
    height * (mode === "drive" ? 0.79 : mode === "ride" ? 0.82 : 0.86);
  const closingFast = track.seenCount >= 3 && track.expansionRate > 0.11;
  const extremeDanger =
    nearBottom &&
    distance <= dangerDistance * 0.58 &&
    (closingFast || ttc <= 1.05);
  let risk: RiskLevel = isInformational ? "info" : "safe";

  if (isHazard && (inPath || (predictedInPath && closingFast))) {
    if (motion.reliable && !thresholds.moving) {
      if (extremeDanger || (closingFast && ttc <= 1.25)) {
        risk = "danger";
      }
    } else {
      const speedSupportsDistanceWarning =
        motion.reliable && motion.speedMps >= 4;
      const distanceWarning =
        distance <= warningDistance &&
        (nearBottom || closingFast || speedSupportsDistanceWarning);

      if (
        extremeDanger ||
        ttc <= profile.dangerTtc ||
        (distance <= dangerDistance &&
          (nearBottom || closingFast || speedSupportsDistanceWarning))
      ) {
        risk = "danger";
      } else if (
        distanceWarning ||
        ttc <= profile.warningTtc ||
        (predictedInPath && closingFast)
      ) {
        risk = "watch";
      }
    }
  }

  const rank =
    risk === "danger" ? 3 : risk === "watch" ? 2 : risk === "safe" ? 1 : 0;
  const severity =
    rank * 100 +
    (inPath ? 24 : predictedInPath ? 12 : 0) +
    Math.max(0, warningDistance - distance) +
    Math.max(0, profile.warningTtc - ttc) * 8;

  return {
    risk,
    ttc,
    inPath,
    predictedInPath,
    severity,
    extremeDanger,
  };
}

function stabilizeRisk(
  track: TrackRecord,
  rawRisk: RiskLevel,
  extremeDanger: boolean,
  mode: TravelMode,
  motion: MotionSnapshot,
) {
  if (rawRisk === "info") {
    const updated = {
      ...track,
      watchEvidence: 0,
      dangerEvidence: 0,
      stableRisk: "info" as const,
    };
    return updated;
  }

  const warningSignal = rawRisk === "watch" || rawRisk === "danger";
  const watchEvidence = warningSignal
    ? Math.min(6, track.watchEvidence + 1)
    : Math.max(0, track.watchEvidence - 1);
  const dangerEvidence =
    rawRisk === "danger"
      ? Math.min(4, track.dangerEvidence + 1)
      : Math.max(0, track.dangerEvidence - 1);
  const fastDrive =
    mode === "drive" && motion.reliable && motion.speedMps >= 12;
  const watchHitsRequired = fastDrive ? 2 : 3;

  let stableRisk: RiskLevel = "safe";
  if (extremeDanger || dangerEvidence >= 2) {
    stableRisk = "danger";
  } else if (watchEvidence >= watchHitsRequired) {
    stableRisk = "watch";
  } else if (track.stableRisk === "danger" && dangerEvidence > 0) {
    stableRisk = "danger";
  } else if (track.stableRisk === "watch" && watchEvidence > 0) {
    stableRisk = "watch";
  }

  return {
    ...track,
    watchEvidence,
    dangerEvidence,
    stableRisk,
  };
}

export function analyzeDetections(
  boxes: Box[],
  width: number,
  height: number,
  mode: TravelMode,
  now: number,
  store: TrackStore,
  motion: MotionSnapshot,
  roadScene?: RoadScene | null,
) {
  const usedTracks = new Set<number>();
  const frameDiagonal = Math.hypot(width, height);
  const detections: AnalyzedDetection[] = [];

  for (const box of boxes) {
    if (detections.length >= 10) break;
    const name = box.name.toLowerCase();
    if (isDashboardDetection(box, width, height, mode, roadScene)) {
      continue;
    }
    if (
      box.conf < 0.35 ||
      box.x2 - box.x1 < width * 0.012 ||
      box.y2 - box.y1 < height * 0.016
    ) {
      continue;
    }

    const previous = matchTrack(
      box,
      now,
      store,
      usedTracks,
      frameDiagonal,
    );
    const track = updateTrack(box, now, previous, store);
    usedTracks.add(track.id);
    const measuredDistance = estimateDistance(box, width, height, mode);
    const distance =
      track.distance === null
        ? measuredDistance
        : track.distance * 0.68 + measuredDistance * 0.32;
    const trackWithDistance = { ...track, distance };
    const assessment = assessRisk(
      box,
      trackWithDistance,
      distance,
      width,
      height,
      mode,
      motion,
      roadScene,
    );
    const stabilizedTrack = stabilizeRisk(
      trackWithDistance,
      assessment.risk,
      assessment.extremeDanger,
      mode,
      motion,
    );
    store.tracks.set(stabilizedTrack.id, stabilizedTrack);
    detections.push({
      ...box,
      id: stabilizedTrack.id,
      thaiName: thaiName(name),
      distance,
      ttc: assessment.ttc,
      risk: stabilizedTrack.stableRisk,
      inPath: assessment.inPath,
      predictedInPath: assessment.predictedInPath,
      direction: directionFor(stabilizedTrack.cx, width),
      severity: assessment.severity,
      history: stabilizedTrack.history,
    });
  }

  for (const [id, track] of store.tracks) {
    if (now - track.updatedAt > 1700) store.tracks.delete(id);
  }

  return detections.sort((a, b) => a.severity - b.severity);
}

export function selectAlert(
  detections: AnalyzedDetection[],
  mode: TravelMode,
  motion: MotionSnapshot,
): VisionAlert | null {
  const candidates = detections
    .filter((detection) => detection.risk === "watch" || detection.risk === "danger")
    .sort((a, b) => b.severity - a.severity);
  const primary = candidates[0];
  if (!primary) return null;

  const direction =
    primary.direction === "กลาง" ? "ด้านหน้า" : `ด้าน${primary.direction}`;
  const danger = primary.risk === "danger";
  const action =
    mode === "walk"
      ? danger
        ? "หยุดหรือหลบเมื่อปลอดภัย"
        : "ชะลอและมองทาง"
      : danger
        ? "ชะลอและเตรียมหยุด"
        : "ลดความเร็วและเพิ่มระยะ";
  const speedDetail = motion.reliable
    ? `ความเร็ว ${formatSpeed(motion.speedMps)} กม./ชม. • `
    : "";

  return {
    key: `${primary.id}-${primary.name}`,
    level: danger ? "danger" : "watch",
    title: danger ? `อันตราย • ${primary.thaiName}${direction}` : `ระวัง • ${primary.thaiName}${direction}`,
    detail: `${action} • ${speedDetail}ระยะประมาณ ${formatDistance(primary.distance)}`,
    objectName: primary.thaiName,
    direction,
    distance: primary.distance,
  };
}

export function formatDistance(distance: number) {
  if (distance < 10) return `${distance.toFixed(1)} ม.`;
  return `${Math.round(distance)} ม.`;
}

function roundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function traceNormalizedPoints(
  context: CanvasRenderingContext2D,
  points: NormalizedPoint[],
  width: number,
  height: number,
) {
  points.forEach((point, index) => {
    const x = point.x * width;
    const y = point.y * height;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
}

function drawDashboardMask(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  dashboard: DashboardMask,
) {
  if (dashboard.points.length < 2) return;
  context.save();
  const fill = context.createLinearGradient(
    0,
    dashboard.topY * height,
    0,
    height,
  );
  fill.addColorStop(0, "rgba(3, 8, 12, 0.08)");
  fill.addColorStop(0.22, "rgba(3, 8, 12, 0.2)");
  fill.addColorStop(1, "rgba(3, 8, 12, 0.38)");

  context.beginPath();
  traceNormalizedPoints(context, dashboard.points, width, height);
  context.lineTo(width, height);
  context.lineTo(0, height);
  context.closePath();
  context.fillStyle = fill;
  context.fill();

  context.beginPath();
  traceNormalizedPoints(context, dashboard.points, width, height);
  context.setLineDash([
    Math.max(8, width * 0.012),
    Math.max(7, width * 0.009),
  ]);
  context.lineWidth = Math.max(1.5, width * 0.0022);
  context.strokeStyle = "rgba(126, 231, 255, 0.52)";
  context.shadowColor = "rgba(126, 231, 255, 0.34)";
  context.shadowBlur = 7;
  context.stroke();
  context.restore();
}

function drawDetectedLane(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  lane: DetectedLane,
  elapsed: number,
  alert: VisionAlert | null,
) {
  const color =
    alert?.level === "danger"
      ? "255, 79, 100"
      : alert?.level === "watch"
        ? "255, 171, 74"
        : "83, 255, 188";

  const top = lane.topY * height;
  const bottom = lane.bottomY * height;
  const fill = context.createLinearGradient(0, top, 0, bottom);
  fill.addColorStop(0, `rgba(${color}, 0)`);
  fill.addColorStop(0.38, `rgba(${color}, 0.045)`);
  fill.addColorStop(1, `rgba(${color}, 0.13)`);

  context.save();
  context.beginPath();
  traceNormalizedPoints(context, lane.left.points, width, height);
  [...lane.right.points].reverse().forEach((point) => {
    context.lineTo(point.x * width, point.y * height);
  });
  context.closePath();
  context.fillStyle = fill;
  context.fill();

  context.lineWidth = Math.max(2.5, width * 0.0042);
  context.lineJoin = "round";
  context.lineCap = "round";
  context.strokeStyle = `rgba(${color}, 0.88)`;
  context.shadowColor = `rgba(${color}, 0.64)`;
  context.shadowBlur = 10;
  context.beginPath();
  traceNormalizedPoints(context, lane.left.points, width, height);
  context.stroke();
  context.beginPath();
  traceNormalizedPoints(context, lane.right.points, width, height);
  context.stroke();
  context.shadowBlur = 0;

  const pulse = (elapsed / 1550) % 1;
  const normalizedY = lane.topY + (lane.bottomY - lane.topY) * pulse;
  const bounds = laneBoundsAt(lane, normalizedY);
  const left = bounds.left * width;
  const right = bounds.right * width;
  const y = normalizedY * height;
  const scan = context.createLinearGradient(left, 0, right, 0);
  scan.addColorStop(0, `rgba(${color}, 0)`);
  scan.addColorStop(0.5, `rgba(${color}, 0.58)`);
  scan.addColorStop(1, `rgba(${color}, 0)`);
  context.fillStyle = scan;
  context.fillRect(left, y, right - left, Math.max(2, height * 0.003));
  context.restore();
}

function drawRoadScene(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  mode: TravelMode,
  elapsed: number,
  roadScene: RoadScene | null,
  alert: VisionAlert | null,
) {
  if (mode !== "drive" || !roadScene) return;
  if (roadScene.dashboard) {
    drawDashboardMask(context, width, height, roadScene.dashboard);
  }
  if (roadScene.lane) {
    drawDetectedLane(
      context,
      width,
      height,
      roadScene.lane,
      elapsed,
      alert,
    );
  }
}

function drawCornerBox(
  context: CanvasRenderingContext2D,
  detection: AnalyzedDetection,
  width: number,
  height: number,
) {
  const color = RISK_COLOR[detection.risk];
  const boxWidth = detection.x2 - detection.x1;
  const boxHeight = detection.y2 - detection.y1;
  const corner = clamp(Math.min(boxWidth, boxHeight) * 0.18, 6, 24);
  const lineWidth = clamp(width / 300, 1.5, 4.5);
  const x1 = clamp(detection.x1, 1, width - 2);
  const y1 = clamp(detection.y1, 1, height - 2);
  const x2 = clamp(detection.x2, 2, width - 1);
  const y2 = clamp(detection.y2, 2, height - 1);

  context.fillStyle =
    detection.risk === "danger"
      ? "rgba(255, 79, 100, 0.16)"
      : detection.risk === "watch"
        ? "rgba(255, 171, 74, 0.1)"
        : "rgba(255, 224, 113, 0.045)";
  context.fillRect(x1, y1, x2 - x1, y2 - y1);

  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.lineCap = "round";
  context.shadowColor = color;
  context.shadowBlur = detection.risk === "danger" ? 13 : 7;
  context.beginPath();
  context.moveTo(x1, y1 + corner);
  context.lineTo(x1, y1);
  context.lineTo(x1 + corner, y1);
  context.moveTo(x2 - corner, y1);
  context.lineTo(x2, y1);
  context.lineTo(x2, y1 + corner);
  context.moveTo(x2, y2 - corner);
  context.lineTo(x2, y2);
  context.lineTo(x2 - corner, y2);
  context.moveTo(x1 + corner, y2);
  context.lineTo(x1, y2);
  context.lineTo(x1, y2 - corner);
  context.stroke();
  context.shadowBlur = 0;

  if (
    (detection.risk === "watch" || detection.risk === "danger") &&
    detection.history.length > 1
  ) {
    context.strokeStyle =
      detection.risk === "danger"
        ? "rgba(255, 79, 100, 0.55)"
        : "rgba(255, 171, 74, 0.45)";
    context.lineWidth = Math.max(1, lineWidth * 0.6);
    context.beginPath();
    detection.history.forEach((point, index) => {
      if (index === 0) context.moveTo(point.x, point.y);
      else context.lineTo(point.x, point.y);
    });
    context.stroke();
  }

  const fontSize = clamp(width / 44, 10, 20);
  context.font = `700 ${fontSize}px "Noto Sans Thai", "Thonburi", sans-serif`;
  const distance = formatDistance(detection.distance);
  const text = `${detection.thaiName}  ${distance}`;
  const paddingX = fontSize * 0.55;
  const labelHeight = fontSize * 1.65;
  const labelWidth = context.measureText(text).width + paddingX * 2;
  const labelX = clamp(x1, 4, width - labelWidth - 4);
  const labelY =
    y1 > labelHeight + 6
      ? y1 - labelHeight - 4
      : Math.min(height - labelHeight - 4, y1 + 4);

  roundedRect(
    context,
    labelX,
    labelY,
    labelWidth,
    labelHeight,
    fontSize * 0.38,
  );
  context.fillStyle =
    detection.risk === "danger"
      ? "rgba(122, 14, 32, 0.94)"
      : detection.risk === "watch"
        ? "rgba(93, 49, 5, 0.93)"
        : detection.risk === "info"
          ? "rgba(3, 56, 70, 0.91)"
          : "rgba(49, 44, 6, 0.9)";
  context.fill();
  context.strokeStyle = color;
  context.lineWidth = Math.max(1, lineWidth * 0.35);
  context.stroke();
  context.fillStyle = "#ffffff";
  context.textBaseline = "middle";
  context.fillText(
    text,
    labelX + paddingX,
    labelY + labelHeight * 0.52,
  );

  context.fillStyle = color;
  context.beginPath();
  context.arc(
    (x1 + x2) / 2,
    y2,
    Math.max(2.5, lineWidth),
    0,
    Math.PI * 2,
  );
  context.fill();
}

export function renderVisionOverlay(
  canvas: HTMLCanvasElement,
  mode: TravelMode,
  elapsed: number,
  detections: AnalyzedDetection[],
  alert: VisionAlert | null,
  roadScene: RoadScene | null,
  sourceWidth: number,
  sourceHeight: number,
) {
  const context = canvas.getContext("2d");
  if (!context) return;
  const outputWidth = canvas.width;
  const outputHeight = canvas.height;
  const width = Math.max(1, sourceWidth);
  const height = Math.max(1, sourceHeight);

  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, outputWidth, outputHeight);
  context.imageSmoothingEnabled = true;
  context.save();
  context.setTransform(
    outputWidth / width,
    0,
    0,
    outputHeight / height,
    0,
    0,
  );
  drawRoadScene(
    context,
    width,
    height,
    mode,
    elapsed,
    roadScene,
    alert,
  );
  detections.forEach((detection) =>
    drawCornerBox(context, detection, width, height),
  );
  context.restore();
}
