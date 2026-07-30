export type NormalizedPoint = {
  x: number;
  y: number;
};

export type LaneBoundary = {
  points: NormalizedPoint[];
};

export type DetectedLane = {
  left: LaneBoundary;
  right: LaneBoundary;
  topY: number;
  bottomY: number;
  confidence: number;
};

export type DashboardMask = {
  points: NormalizedPoint[];
  topY: number;
  confidence: number;
};

export type RoadScene = {
  lane: DetectedLane | null;
  dashboard: DashboardMask | null;
  analyzedAt: number;
};

export type RoadSceneTracker = {
  lane: DetectedLane | null;
  laneEvidence: number;
  dashboard: DashboardMask | null;
  dashboardEvidence: number;
};

type EdgePoint = {
  x: number;
  y: number;
  weight: number;
  paint: number;
};

type FittedLine = {
  slope: number;
  intercept: number;
  confidence: number;
  support: EdgePoint[];
};

const EMPTY_SCENE: RoadScene = {
  lane: null,
  dashboard: null,
  analyzedAt: 0,
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function interpolate(
  start: number,
  end: number,
  amount: number,
) {
  return start + (end - start) * amount;
}

export function createRoadSceneTracker(): RoadSceneTracker {
  return {
    lane: null,
    laneEvidence: 0,
    dashboard: null,
    dashboardEvidence: 0,
  };
}

export function emptyRoadScene(analyzedAt = 0): RoadScene {
  return { ...EMPTY_SCENE, analyzedAt };
}

function luminanceBuffer(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
) {
  const luminance = new Float32Array(width * height);
  for (let index = 0; index < luminance.length; index += 1) {
    const pixelIndex = index * 4;
    luminance[index] =
      pixels[pixelIndex] * 0.299 +
      pixels[pixelIndex + 1] * 0.587 +
      pixels[pixelIndex + 2] * 0.114;
  }
  return luminance;
}

function detectDashboard(
  luminance: Float32Array,
  width: number,
  height: number,
): DashboardMask | null {
  const startX = Math.max(2, Math.floor(width * 0.055));
  const endX = Math.min(width - 3, Math.ceil(width * 0.945));
  const startY = Math.max(3, Math.floor(height * 0.56));
  const endY = Math.min(height - 4, Math.ceil(height * 0.91));
  const rowStep = width >= 260 ? 2 : 1;
  let bestY = -1;
  let bestScore = 0;
  let bestCoverage = 0;
  let bestMean = 0;

  for (let y = startY; y <= endY; y += 1) {
    let gradientTotal = 0;
    let strongCount = 0;
    let samples = 0;
    for (let x = startX; x <= endX; x += rowStep) {
      const gradient = Math.abs(
        luminance[(y + 2) * width + x] -
          luminance[(y - 2) * width + x],
      );
      gradientTotal += Math.min(70, gradient);
      if (gradient >= 18) strongCount += 1;
      samples += 1;
    }

    const mean = gradientTotal / Math.max(1, samples);
    const coverage = strongCount / Math.max(1, samples);
    const lowerPreference =
      0.86 + ((y - startY) / Math.max(1, endY - startY)) * 0.18;
    const score = (mean * 0.58 + coverage * 34) * lowerPreference;
    if (score > bestScore) {
      bestScore = score;
      bestY = y;
      bestCoverage = coverage;
      bestMean = mean;
    }
  }

  if (
    bestY < 0 ||
    bestScore < 12.5 ||
    bestCoverage < 0.16 ||
    bestMean < 6.2
  ) {
    return null;
  }

  const anchorCount = 7;
  const searchRadius = Math.max(4, Math.round(height * 0.045));
  const horizontalRadius = Math.max(2, Math.round(width * 0.018));
  const anchors: NormalizedPoint[] = [];
  const anchorStrengths: number[] = [];

  for (let anchor = 0; anchor < anchorCount; anchor += 1) {
    const normalizedX = anchor / (anchorCount - 1);
    const centerX = clamp(
      Math.round(normalizedX * (width - 1)),
      horizontalRadius,
      width - horizontalRadius - 1,
    );
    let strongestY = bestY;
    let strongestGradient = 0;

    for (
      let y = Math.max(3, bestY - searchRadius);
      y <= Math.min(height - 4, bestY + searchRadius);
      y += 1
    ) {
      let total = 0;
      let samples = 0;
      for (
        let x = centerX - horizontalRadius;
        x <= centerX + horizontalRadius;
        x += 1
      ) {
        total += Math.abs(
          luminance[(y + 2) * width + x] -
            luminance[(y - 2) * width + x],
        );
        samples += 1;
      }
      const gradient = total / Math.max(1, samples);
      if (gradient > strongestGradient) {
        strongestGradient = gradient;
        strongestY = y;
      }
    }

    const trustedY =
      strongestGradient >= Math.max(8, bestMean * 0.72)
        ? strongestY
        : bestY;
    anchors.push({
      x: normalizedX,
      y: trustedY / height,
    });
    anchorStrengths.push(strongestGradient);
  }

  const sortedY = anchors.map((point) => point.y).sort((a, b) => a - b);
  const medianY = sortedY[Math.floor(sortedY.length / 2)] ?? bestY / height;
  const consistentAnchors = anchors.filter(
    (point) => Math.abs(point.y - medianY) <= 0.075,
  );
  if (consistentAnchors.length < 5) return null;

  const smoothed = anchors.map((point, index, points) => {
    const previous = points[Math.max(0, index - 1)].y;
    const next = points[Math.min(points.length - 1, index + 1)].y;
    return {
      x: point.x,
      y: clamp(previous * 0.2 + point.y * 0.6 + next * 0.2, 0.54, 0.94),
    };
  });
  const strength =
    anchorStrengths.reduce((total, value) => total + value, 0) /
    anchorStrengths.length;
  const confidence = clamp(
    (bestScore - 10) / 25 +
      bestCoverage * 0.35 +
      clamp((strength - 8) / 35, 0, 0.25),
    0,
    1,
  );

  if (confidence < 0.34) return null;
  return {
    points: smoothed,
    topY: medianY,
    confidence,
  };
}

function collectLaneEdges(
  pixels: Uint8ClampedArray,
  luminance: Float32Array,
  width: number,
  height: number,
  topY: number,
  bottomY: number,
) {
  const points: EdgePoint[] = [];
  const startY = Math.max(2, Math.floor(topY * height));
  const endY = Math.min(height - 3, Math.ceil(bottomY * height));
  const step = Math.max(width, height) >= 300 ? 2 : 1;

  for (let y = startY; y <= endY; y += step) {
    const normalizedY = y / height;
    const perspectiveHalf = interpolate(
      0.18,
      0.57,
      clamp((normalizedY - topY) / Math.max(0.01, bottomY - topY), 0, 1),
    );
    const startX = Math.max(2, Math.floor(width * (0.5 - perspectiveHalf)));
    const endX = Math.min(
      width - 3,
      Math.ceil(width * (0.5 + perspectiveHalf)),
    );

    for (let x = startX; x <= endX; x += step) {
      const center = y * width + x;
      const gx =
        -luminance[center - width - 1] +
        luminance[center - width + 1] -
        luminance[center - 1] * 2 +
        luminance[center + 1] * 2 -
        luminance[center + width - 1] +
        luminance[center + width + 1];
      const gy =
        -luminance[center - width - 1] -
        luminance[center - width] * 2 -
        luminance[center - width + 1] +
        luminance[center + width - 1] +
        luminance[center + width] * 2 +
        luminance[center + width + 1];
      const magnitude = Math.hypot(gx, gy);
      if (magnitude < 64 || Math.abs(gx) < Math.abs(gy) * 0.18) continue;

      const pixelIndex = center * 4;
      const red = pixels[pixelIndex];
      const green = pixels[pixelIndex + 1];
      const blue = pixels[pixelIndex + 2];
      const maximum = Math.max(red, green, blue);
      const minimum = Math.min(red, green, blue);
      const white =
        luminance[center] >= 138 && maximum - minimum <= 58;
      const yellow =
        red >= 118 &&
        green >= 92 &&
        blue <= Math.min(red, green) * 0.82;
      const paint = white || yellow ? 1 : 0;
      points.push({
        x: x / width,
        y: normalizedY,
        weight:
          clamp(magnitude / 110, 0.55, 3.2) * (paint ? 1.65 : 1),
        paint,
      });
    }
  }
  return points;
}

function fitLaneLine(
  points: EdgePoint[],
  side: "left" | "right",
  topY: number,
  bottomY: number,
): FittedLine | null {
  const slopeCount = 35;
  const interceptMinimum = -1.45;
  const interceptMaximum = 2.45;
  const interceptBins = 196;
  const accumulator = new Float32Array(slopeCount * interceptBins);
  const sign = side === "left" ? -1 : 1;

  for (const point of points) {
    if (
      (side === "left" && point.x > 0.59) ||
      (side === "right" && point.x < 0.41)
    ) {
      continue;
    }
    for (let slopeIndex = 0; slopeIndex < slopeCount; slopeIndex += 1) {
      const magnitude = interpolate(
        0.075,
        1.38,
        slopeIndex / (slopeCount - 1),
      );
      const slope = magnitude * sign;
      const intercept = point.x - slope * point.y;
      const interceptIndex = Math.floor(
        ((intercept - interceptMinimum) /
          (interceptMaximum - interceptMinimum)) *
          interceptBins,
      );
      if (interceptIndex < 0 || interceptIndex >= interceptBins) continue;
      accumulator[slopeIndex * interceptBins + interceptIndex] +=
        point.weight;
    }
  }

  let bestIndex = -1;
  let bestVotes = 0;
  for (let index = 0; index < accumulator.length; index += 1) {
    if (accumulator[index] > bestVotes) {
      bestVotes = accumulator[index];
      bestIndex = index;
    }
  }
  if (bestIndex < 0 || bestVotes < 13) return null;

  const slopeIndex = Math.floor(bestIndex / interceptBins);
  const interceptIndex = bestIndex % interceptBins;
  let slope =
    interpolate(0.075, 1.38, slopeIndex / (slopeCount - 1)) * sign;
  let intercept =
    interceptMinimum +
    ((interceptIndex + 0.5) / interceptBins) *
      (interceptMaximum - interceptMinimum);
  let support = points.filter((point) => {
    if (
      (side === "left" && point.x > 0.61) ||
      (side === "right" && point.x < 0.39)
    ) {
      return false;
    }
    return Math.abs(point.x - (slope * point.y + intercept)) <= 0.022;
  });
  if (support.length < 9) return null;

  let totalWeight = 0;
  let sumY = 0;
  let sumX = 0;
  let sumYY = 0;
  let sumYX = 0;
  for (const point of support) {
    totalWeight += point.weight;
    sumY += point.y * point.weight;
    sumX += point.x * point.weight;
    sumYY += point.y * point.y * point.weight;
    sumYX += point.y * point.x * point.weight;
  }
  const denominator = totalWeight * sumYY - sumY * sumY;
  if (Math.abs(denominator) < 0.00001) return null;
  slope = (totalWeight * sumYX - sumY * sumX) / denominator;
  intercept = (sumX - slope * sumY) / totalWeight;

  if (
    (side === "left" && (slope >= -0.055 || slope < -1.55)) ||
    (side === "right" && (slope <= 0.055 || slope > 1.55))
  ) {
    return null;
  }

  support = points.filter(
    (point) =>
      Math.abs(point.x - (slope * point.y + intercept)) <= 0.02,
  );
  if (support.length < 9) return null;

  const coveredBands = new Set<number>();
  let residualTotal = 0;
  let weightedSupport = 0;
  let paintSupport = 0;
  for (const point of support) {
    coveredBands.add(
      clamp(
        Math.floor(
          ((point.y - topY) / Math.max(0.01, bottomY - topY)) * 12,
        ),
        0,
        11,
      ),
    );
    residualTotal +=
      Math.abs(point.x - (slope * point.y + intercept)) * point.weight;
    weightedSupport += point.weight;
    paintSupport += point.paint * point.weight;
  }
  const coverage = coveredBands.size / 12;
  const residual = residualTotal / Math.max(1, weightedSupport);
  const paintRatio = paintSupport / Math.max(1, weightedSupport);
  const confidence = clamp(
    (coverage - 0.16) * 1.45 +
      clamp((weightedSupport - 15) / 95, 0, 0.35) +
      clamp((0.018 - residual) / 0.018, 0, 0.25) +
      paintRatio * 0.2,
    0,
    1,
  );
  if (coverage < 0.24 || residual > 0.018 || confidence < 0.36) {
    return null;
  }

  return { slope, intercept, confidence, support };
}

function traceBoundary(
  line: FittedLine,
  topY: number,
  bottomY: number,
) {
  const points: NormalizedPoint[] = [];
  const samples = 7;
  for (let index = 0; index < samples; index += 1) {
    const y = interpolate(topY, bottomY, index / (samples - 1));
    const predictedX = line.slope * y + line.intercept;
    const nearby = line.support.filter(
      (point) =>
        Math.abs(point.y - y) <= 0.045 &&
        Math.abs(point.x - predictedX) <= 0.06,
    );
    let observedX = predictedX;
    if (nearby.length) {
      let total = 0;
      let weightedX = 0;
      for (const point of nearby) {
        const proximity =
          1 - Math.abs(point.y - y) / 0.045;
        const weight = point.weight * Math.max(0.15, proximity);
        total += weight;
        weightedX += point.x * weight;
      }
      observedX = weightedX / Math.max(0.001, total);
    }
    points.push({
      x: clamp(
        predictedX * 0.46 +
          clamp(observedX, predictedX - 0.055, predictedX + 0.055) *
            0.54,
        -0.05,
        1.05,
      ),
      y,
    });
  }
  return points;
}

function detectLane(
  pixels: Uint8ClampedArray,
  luminance: Float32Array,
  width: number,
  height: number,
  dashboard: DashboardMask | null,
): DetectedLane | null {
  const roiTop = 0.34;
  const roadBottom = clamp(
    dashboard ? dashboard.topY - 0.012 : 0.955,
    0.68,
    0.965,
  );
  if (roadBottom - roiTop < 0.3) return null;

  const edgePoints = collectLaneEdges(
    pixels,
    luminance,
    width,
    height,
    roiTop,
    roadBottom,
  );
  if (edgePoints.length < 28) return null;
  const left = fitLaneLine(edgePoints, "left", roiTop, roadBottom);
  const right = fitLaneLine(edgePoints, "right", roiTop, roadBottom);
  if (!left || !right) return null;

  const denominator = left.slope - right.slope;
  if (Math.abs(denominator) < 0.12) return null;
  const vanishY = (right.intercept - left.intercept) / denominator;
  const vanishX = left.slope * vanishY + left.intercept;
  if (
    vanishY < 0.16 ||
    vanishY > 0.67 ||
    vanishX < 0.2 ||
    vanishX > 0.8
  ) {
    return null;
  }

  const topY = clamp(
    vanishY + 0.055,
    0.35,
    Math.min(0.62, roadBottom - 0.22),
  );
  const leftTop = left.slope * topY + left.intercept;
  const rightTop = right.slope * topY + right.intercept;
  const leftBottom = left.slope * roadBottom + left.intercept;
  const rightBottom = right.slope * roadBottom + right.intercept;
  const topWidth = rightTop - leftTop;
  const bottomWidth = rightBottom - leftBottom;
  const bottomCenter = (leftBottom + rightBottom) / 2;

  if (
    topWidth < 0.025 ||
    topWidth > 0.44 ||
    bottomWidth < 0.2 ||
    bottomWidth > 1.04 ||
    bottomCenter < 0.2 ||
    bottomCenter > 0.8 ||
    leftBottom >= rightBottom
  ) {
    return null;
  }

  const perspectiveGrowth = bottomWidth / Math.max(0.02, topWidth);
  const geometryConfidence = clamp(
    1 -
      Math.abs(vanishX - 0.5) * 0.7 -
      Math.abs(bottomCenter - 0.5) * 0.55,
    0,
    1,
  );
  if (perspectiveGrowth < 1.35 || geometryConfidence < 0.46) return null;

  const confidence = clamp(
    Math.min(left.confidence, right.confidence) * 0.72 +
      geometryConfidence * 0.28,
    0,
    1,
  );
  if (confidence < 0.43) return null;
  return {
    left: { points: traceBoundary(left, topY, roadBottom) },
    right: { points: traceBoundary(right, topY, roadBottom) },
    topY,
    bottomY: roadBottom,
    confidence,
  };
}

function averagePointDifference(
  current: NormalizedPoint[],
  previous: NormalizedPoint[],
) {
  const count = Math.min(current.length, previous.length);
  if (!count) return 1;
  let difference = 0;
  for (let index = 0; index < count; index += 1) {
    difference += Math.hypot(
      current[index].x - previous[index].x,
      current[index].y - previous[index].y,
    );
  }
  return difference / count;
}

function blendPoints(
  current: NormalizedPoint[],
  previous: NormalizedPoint[],
  currentWeight: number,
) {
  return current.map((point, index) => {
    const oldPoint = previous[index] ?? point;
    return {
      x: interpolate(oldPoint.x, point.x, currentWeight),
      y: interpolate(oldPoint.y, point.y, currentWeight),
    };
  });
}

function stabilizeDashboard(
  raw: DashboardMask | null,
  tracker: RoadSceneTracker,
) {
  if (!raw) {
    tracker.dashboard = null;
    tracker.dashboardEvidence = 0;
    return null;
  }

  const previous = tracker.dashboard;
  const consistent =
    previous &&
    Math.abs(previous.topY - raw.topY) <= 0.065 &&
    averagePointDifference(raw.points, previous.points) <= 0.085;
  tracker.dashboardEvidence = consistent
    ? Math.min(5, tracker.dashboardEvidence + 1)
    : 1;
  tracker.dashboard = consistent
    ? {
        points: blendPoints(raw.points, previous.points, 0.38),
        topY: interpolate(previous.topY, raw.topY, 0.38),
        confidence: interpolate(previous.confidence, raw.confidence, 0.45),
      }
    : raw;
  return tracker.dashboardEvidence >= 2 ? tracker.dashboard : null;
}

function stabilizeLane(
  raw: DetectedLane | null,
  tracker: RoadSceneTracker,
) {
  if (!raw) {
    tracker.lane = null;
    tracker.laneEvidence = 0;
    return null;
  }

  const previous = tracker.lane;
  const difference = previous
    ? (averagePointDifference(raw.left.points, previous.left.points) +
        averagePointDifference(raw.right.points, previous.right.points)) /
      2
    : 1;
  const consistent =
    previous &&
    difference <= 0.105 &&
    Math.abs(previous.bottomY - raw.bottomY) <= 0.075;
  tracker.laneEvidence = consistent
    ? Math.min(6, tracker.laneEvidence + 1)
    : 1;
  tracker.lane = consistent
    ? {
        left: {
          points: blendPoints(raw.left.points, previous.left.points, 0.4),
        },
        right: {
          points: blendPoints(raw.right.points, previous.right.points, 0.4),
        },
        topY: interpolate(previous.topY, raw.topY, 0.4),
        bottomY: interpolate(previous.bottomY, raw.bottomY, 0.4),
        confidence: interpolate(previous.confidence, raw.confidence, 0.45),
      }
    : raw;
  return tracker.laneEvidence >= 2 ? tracker.lane : null;
}

export function analyzeRoadScene(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  tracker: RoadSceneTracker,
  analyzedAt: number,
): RoadScene {
  if (
    width < 96 ||
    height < 96 ||
    pixels.length < width * height * 4
  ) {
    tracker.lane = null;
    tracker.dashboard = null;
    tracker.laneEvidence = 0;
    tracker.dashboardEvidence = 0;
    return emptyRoadScene(analyzedAt);
  }

  const luminance = luminanceBuffer(pixels, width, height);
  const rawDashboard = detectDashboard(luminance, width, height);
  const dashboard = stabilizeDashboard(rawDashboard, tracker);
  const rawLane = detectLane(
    pixels,
    luminance,
    width,
    height,
    rawDashboard ?? dashboard,
  );
  const lane = stabilizeLane(rawLane, tracker);
  return { lane, dashboard, analyzedAt };
}

function interpolateAtAxis(
  points: NormalizedPoint[],
  value: number,
  axis: "x" | "y",
) {
  if (!points.length) return 0.5;
  const otherAxis = axis === "x" ? "y" : "x";
  if (value <= points[0][axis]) return points[0][otherAxis];
  const last = points[points.length - 1];
  if (value >= last[axis]) return last[otherAxis];

  for (let index = 1; index < points.length; index += 1) {
    const current = points[index];
    const previous = points[index - 1];
    if (value > current[axis]) continue;
    const progress =
      (value - previous[axis]) /
      Math.max(0.0001, current[axis] - previous[axis]);
    return interpolate(previous[otherAxis], current[otherAxis], progress);
  }
  return last[otherAxis];
}

export function laneBoundsAt(
  lane: DetectedLane,
  normalizedY: number,
) {
  return {
    left: interpolateAtAxis(lane.left.points, normalizedY, "y"),
    right: interpolateAtAxis(lane.right.points, normalizedY, "y"),
  };
}

export function dashboardTopAt(
  dashboard: DashboardMask,
  normalizedX: number,
) {
  return interpolateAtAxis(dashboard.points, normalizedX, "x");
}
