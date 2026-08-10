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
  laneMisses: number;
  dashboard: DashboardMask | null;
  dashboardEvidence: number;
  scratchPixels: Uint8ClampedArray | null;
  scratchLuminance: Float32Array | null;
};

type EdgePoint = {
  x: number;
  y: number;
  weight: number;
  paint: number;
  white: number;
  yellow: number;
};

type LanePaintType = "white" | "yellow";

type FittedLine = {
  slope: number;
  intercept: number;
  confidence: number;
  support: EdgePoint[];
  paintType: LanePaintType;
};

type TracedBoundary = {
  points: NormalizedPoint[];
  observed: boolean[];
  confidence: number;
  observedRatio: number;
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
    laneMisses: 0,
    dashboard: null,
    dashboardEvidence: 0,
    scratchPixels: null,
    scratchLuminance: null,
  };
}

function paintScoresAt(
  pixels: Uint8ClampedArray,
  luminance: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
) {
  const leftX = Math.max(0, x - 7);
  const rightX = Math.min(width - 1, x + 7);
  const upperY = Math.max(0, y - 4);
  const lowerY = Math.min(height - 1, y + 4);
  const background =
    (luminance[y * width + leftX] +
      luminance[y * width + rightX] +
      luminance[upperY * width + x] +
      luminance[lowerY * width + x]) /
    4;
  let white = 0;
  let yellow = 0;

  // An edge sample can land on either side of a painted stripe. Inspect a
  // narrow cross-section and keep the most paint-like pixel near the edge.
  for (let offset = -2; offset <= 2; offset += 1) {
    const sampleX = clamp(x + offset, 0, width - 1);
    const center = y * width + sampleX;
    const pixelIndex = center * 4;
    const red = pixels[pixelIndex];
    const green = pixels[pixelIndex + 1];
    const blue = pixels[pixelIndex + 2];
    const maximum = Math.max(red, green, blue);
    const minimum = Math.min(red, green, blue);
    const chroma = maximum - minimum;
    const brightness = luminance[center];
    const contrast = brightness - background;

    const neutral = 1 - clamp((chroma - 12) / 78, 0, 1);
    const whiteBrightness = clamp((brightness - 88) / 112, 0, 1);
    const whiteContrast = clamp((contrast - 4) / 42, 0, 1);
    white = Math.max(
      white,
      neutral *
        whiteBrightness *
        (0.28 + whiteContrast * 0.72),
    );

    const warmSeparation = Math.min(red, green) - blue;
    const yellowHue =
      clamp((warmSeparation - 5) / 72, 0, 1) *
      clamp((green - red * 0.46) / 64, 0, 1);
    const yellowBrightness = clamp((brightness - 58) / 125, 0, 1);
    const yellowContrast = clamp((contrast - 2) / 38, 0, 1);
    yellow = Math.max(
      yellow,
      yellowHue *
        yellowBrightness *
        (0.32 + yellowContrast * 0.68),
    );
  }

  return { white, yellow, paint: Math.max(white, yellow) };
}

export function emptyRoadScene(analyzedAt = 0): RoadScene {
  return { ...EMPTY_SCENE, analyzedAt };
}

function luminanceBuffer(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  reusable: Float32Array | null,
) {
  const requiredLength = width * height;
  const luminance =
    reusable?.length === requiredLength
      ? reusable
      : new Float32Array(requiredLength);
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
      if (magnitude < 48 || Math.abs(gx) < Math.abs(gy) * 0.14) continue;

      const scores = paintScoresAt(
        pixels,
        luminance,
        width,
        height,
        x,
        y,
      );
      // Only painted white/yellow candidates enter the lane tracker. A
      // dashed gap is extrapolated from the lane path instead of being
      // filled with an unrelated vehicle, barrier, shadow, or asphalt edge.
      if (scores.paint < 0.16) continue;
      points.push({
        x: x / width,
        y: normalizedY,
        weight:
          clamp(magnitude / 125, 0.45, 3) *
          (0.38 + scores.paint * 1.62),
        paint: scores.paint,
        white: scores.white,
        yellow: scores.yellow,
      });
    }
  }
  return points;
}

function fitLaneLines(
  points: EdgePoint[],
  side: "left" | "right",
  topY: number,
  bottomY: number,
): FittedLine[] {
  const slopeCount = 35;
  const interceptMinimum = -1.45;
  const interceptMaximum = 2.45;
  const interceptBins = 196;
  const accumulator = new Float32Array(slopeCount * interceptBins);
  const sign = side === "left" ? -1 : 1;

  for (const point of points) {
    if (
      (side === "left" && point.x > 0.64) ||
      (side === "right" && point.x < 0.36)
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

  const ranked: Array<{
    slopeIndex: number;
    interceptIndex: number;
    score: number;
  }> = [];
  for (let index = 0; index < accumulator.length; index += 1) {
    const votes = accumulator[index];
    if (votes < 6) continue;
    const slopeIndex = Math.floor(index / interceptBins);
    const interceptIndex = index % interceptBins;
    const slope =
      interpolate(0.075, 1.38, slopeIndex / (slopeCount - 1)) * sign;
    const intercept =
      interceptMinimum +
      ((interceptIndex + 0.5) / interceptBins) *
        (interceptMaximum - interceptMinimum);
    const bottomX = slope * bottomY + intercept;
    const centerDistance =
      side === "left" ? 0.5 - bottomX : bottomX - 0.5;
    if (centerDistance < 0.035 || centerDistance > 0.72) continue;
    // The current lane is bounded by the first reliable marking on either
    // side of the camera center. This preference stops a large white car or
    // a farther adjacent-lane stripe from winning purely on pixel count.
    const centerPreference =
      1 / Math.pow(centerDistance + 0.065, 2);
    ranked.push({
      slopeIndex,
      interceptIndex,
      score: votes * centerPreference,
    });
  }
  ranked.sort((first, second) => second.score - first.score);
  const candidates: typeof ranked = [];
  for (const candidate of ranked) {
    const duplicate = candidates.some(
      (chosen) =>
        Math.abs(chosen.slopeIndex - candidate.slopeIndex) <= 2 &&
        Math.abs(chosen.interceptIndex - candidate.interceptIndex) <= 4,
    );
    if (!duplicate) candidates.push(candidate);
    if (candidates.length >= 18) break;
  }
  const fitted: FittedLine[] = [];

  for (const candidate of candidates) {
    let slope =
      interpolate(
        0.075,
        1.38,
        candidate.slopeIndex / (slopeCount - 1),
      ) * sign;
    let intercept =
      interceptMinimum +
      ((candidate.interceptIndex + 0.5) / interceptBins) *
        (interceptMaximum - interceptMinimum);
    let support = points.filter((point) => {
      if (
        (side === "left" && point.x > 0.66) ||
        (side === "right" && point.x < 0.34)
      ) {
        return false;
      }
      return Math.abs(point.x - (slope * point.y + intercept)) <= 0.022;
    });
    if (support.length < 7) continue;

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
    if (Math.abs(denominator) < 0.00001) continue;
    slope = (totalWeight * sumYX - sumY * sumX) / denominator;
    intercept = (sumX - slope * sumY) / totalWeight;

    if (
      (side === "left" && (slope >= -0.055 || slope < -1.55)) ||
      (side === "right" && (slope <= 0.055 || slope > 1.55))
    ) {
      continue;
    }
    const refinedBottomX = slope * bottomY + intercept;
    const refinedDistance =
      side === "left"
        ? 0.5 - refinedBottomX
        : refinedBottomX - 0.5;
    if (refinedDistance < 0.035 || refinedDistance > 0.72) continue;

    support = points.filter(
      (point) =>
        Math.abs(point.x - (slope * point.y + intercept)) <= 0.024,
    );
    if (support.length < 7) continue;
    let minimumSupportY = 1;
    let maximumSupportY = 0;
    for (const point of support) {
      minimumSupportY = Math.min(minimumSupportY, point.y);
      maximumSupportY = Math.max(maximumSupportY, point.y);
    }
    const roiHeight = bottomY - topY;
    if (
      maximumSupportY - minimumSupportY < roiHeight * 0.44 ||
      maximumSupportY < topY + roiHeight * 0.74
    ) {
      continue;
    }

    const coveredBands = new Set<number>();
    let residualTotal = 0;
    let weightedSupport = 0;
    let paintSupport = 0;
    let whiteSupport = 0;
    let yellowSupport = 0;
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
      whiteSupport += point.white * point.weight;
      yellowSupport += point.yellow * point.weight;
    }
    const coverage = coveredBands.size / 12;
    const residual = residualTotal / Math.max(1, weightedSupport);
    const paintRatio = paintSupport / Math.max(1, weightedSupport);
    const confidence = clamp(
      (coverage - 0.16) * 1.45 +
        clamp((weightedSupport - 15) / 95, 0, 0.35) +
        clamp((0.024 - residual) / 0.024, 0, 0.25) +
        paintRatio * 0.2,
      0,
      1,
    );
    if (
      coverage < 0.25 ||
      paintRatio < 0.2 ||
      residual > 0.024 ||
      confidence < 0.29
    ) {
      continue;
    }

    fitted.push({
      slope,
      intercept,
      confidence,
      support,
      paintType: yellowSupport > whiteSupport ? "yellow" : "white",
    });
    if (fitted.length >= 8) break;
  }

  return fitted;
}

function traceBoundaryFromImage(
  allEdges: EdgePoint[],
  seed: FittedLine,
  side: "left" | "right",
  topY: number,
  bottomY: number,
  previous: LaneBoundary | null,
): TracedBoundary {
  // The straight Hough line is only a coarse starting point. From there,
  // follow the strongest connected edge band-by-band from the foreground
  // towards the horizon, allowing the observed boundary to bend away from
  // the seed. This preserves real curves instead of drawing the seed line.
  const samples = 13;
  const traced = new Array<NormalizedPoint>(samples);
  const observed = new Array<boolean>(samples).fill(false);
  let observedBands = 0;
  let paintedBands = 0;
  let previousOffset = 0;
  let offsetVelocity = 0;

  for (let index = samples - 1; index >= 0; index -= 1) {
    const progress = index / (samples - 1);
    const y = interpolate(topY, bottomY, progress);
    const seedX = seed.slope * y + seed.intercept;
    const previousX = previous
      ? interpolateAtAxis(previous.points, y, "y")
      : seedX;
    const referenceX = previous
      ? interpolate(seedX, previousX, 0.62)
      : seedX;
    const predictedOffset = previousOffset + offsetVelocity * 0.62;
    const predictedX = referenceX + predictedOffset;
    const yRadius = interpolate(0.026, 0.044, progress);
    const searchRadius = interpolate(0.052, 0.098, progress);
    const clusterRadius = interpolate(0.014, 0.027, progress);
    const candidates = allEdges.filter((point) => {
      if (Math.abs(point.y - y) > yRadius) return false;
      if (Math.abs(point.x - predictedX) > searchRadius) return false;
      if (side === "left" && point.x > 0.69) return false;
      if (side === "right" && point.x < 0.31) return false;
      const colorScore =
        seed.paintType === "yellow"
          ? Math.max(point.yellow, point.white * 0.42)
          : Math.max(point.white, point.yellow * 0.42);
      return colorScore >= 0.12;
    });

    let observedX: number | null = null;
    let observedPaint = 0;
    if (candidates.length) {
      const binCount = 48;
      const densityBins = new Float32Array(binCount);
      const searchStart = predictedX - searchRadius;
      const searchWidth = searchRadius * 2;
      for (const point of candidates) {
        const bin = clamp(
          Math.floor(((point.x - searchStart) / searchWidth) * binCount),
          0,
          binCount - 1,
        );
        const verticalProximity =
          1 - Math.abs(point.y - y) / Math.max(0.001, yRadius);
        const predictionProximity =
          1 -
          Math.abs(point.x - predictedX) /
            Math.max(0.001, searchRadius);
        const colorScore =
          seed.paintType === "yellow"
            ? Math.max(point.yellow, point.white * 0.42)
            : Math.max(point.white, point.yellow * 0.42);
        densityBins[bin] +=
          point.weight *
          (0.35 + colorScore * 1.65) *
          Math.max(0.12, verticalProximity) *
          Math.max(0.16, predictionProximity);
      }

      let bestBin = -1;
      let bestDensity = 0;
      for (let bin = 0; bin < binCount; bin += 1) {
        const density =
          densityBins[bin] +
          (densityBins[bin - 1] ?? 0) * 0.55 +
          (densityBins[bin + 1] ?? 0) * 0.55;
        if (density > bestDensity) {
          bestDensity = density;
          bestBin = bin;
        }
      }

      if (bestBin >= 0 && bestDensity >= 1.25) {
        const anchorX =
          searchStart + ((bestBin + 0.5) / binCount) * searchWidth;
        let total = 0;
        let weightedX = 0;
        let paintTotal = 0;
        for (const point of candidates) {
          const horizontalDistance = Math.abs(point.x - anchorX);
          if (horizontalDistance > clusterRadius) continue;
          const weight =
            point.weight *
            Math.max(0.15, 1 - horizontalDistance / clusterRadius);
          total += weight;
          weightedX += point.x * weight;
          paintTotal += point.paint * weight;
        }
        if (total > 0) {
          observedX = weightedX / total;
          observedPaint = paintTotal / total;
        }
      }
    }

    let nextOffset = predictedOffset;
    if (observedX !== null) {
      observedBands += 1;
      observed[index] = true;
      if (observedPaint >= 0.34) paintedBands += 1;
      const measuredOffset = clamp(
        observedX - referenceX,
        -searchRadius,
        searchRadius,
      );
      const observationWeight = observedPaint >= 0.34 ? 0.72 : 0.56;
      nextOffset = interpolate(
        predictedOffset,
        measuredOffset,
        observationWeight,
      );
    } else {
      // Missing paint or a dashed line can create a short gap. Continue the
      // local curve through the gap, but gently return towards the seed.
      nextOffset = predictedOffset * 0.96;
    }

    offsetVelocity = clamp(nextOffset - previousOffset, -0.028, 0.028);
    previousOffset = nextOffset;
    traced[index] = {
      x: clamp(referenceX + nextOffset, -0.06, 1.06),
      y,
    };
  }

  // A light three-point filter removes single-pixel jitter while retaining
  // the multi-band curve captured from the current camera frame.
  const points = traced.map((point, index, source) => {
    if (index === 0 || index === source.length - 1) return point;
    return {
      x:
        source[index - 1].x * 0.18 +
        point.x * 0.64 +
        source[index + 1].x * 0.18,
      y: point.y,
    };
  });
  const observedRatio = observedBands / samples;
  const paintRatio = paintedBands / Math.max(1, observedBands);
  const confidence = clamp(
    seed.confidence * 0.38 +
      observedRatio * 0.47 +
      paintRatio * 0.15,
    0,
    1,
  );
  return { points, observed, confidence, observedRatio };
}

function regularizeLanePair(
  left: TracedBoundary,
  right: TracedBoundary,
  leftSeed: FittedLine,
  rightSeed: FittedLine,
  previous: DetectedLane | null,
) {
  const leftPoints: NormalizedPoint[] = [];
  const rightPoints: NormalizedPoint[] = [];

  for (let index = 0; index < left.points.length; index += 1) {
    const y = left.points[index].y;
    const progress = index / Math.max(1, left.points.length - 1);
    const seedLeft = leftSeed.slope * y + leftSeed.intercept;
    const seedRight = rightSeed.slope * y + rightSeed.intercept;
    const seedCenter = (seedLeft + seedRight) / 2;
    const seedWidth = Math.max(0.024, seedRight - seedLeft);
    const previousBounds = previous ? laneBoundsAt(previous, y) : null;
    const expectedCenter = previousBounds
      ? interpolate(
          seedCenter,
          (previousBounds.left + previousBounds.right) / 2,
          0.68,
        )
      : seedCenter;
    const expectedWidth = previousBounds
      ? interpolate(
          seedWidth,
          previousBounds.right - previousBounds.left,
          0.68,
        )
      : seedWidth;
    const leftObserved = left.observed[index];
    const rightObserved = right.observed[index];
    let center = expectedCenter;
    let widthAtBand = expectedWidth;

    if (leftObserved && rightObserved) {
      const measuredCenter =
        (left.points[index].x + right.points[index].x) / 2;
      const measuredWidth =
        right.points[index].x - left.points[index].x;
      const centerTolerance = interpolate(0.045, 0.085, progress);
      center = clamp(
        measuredCenter,
        expectedCenter - centerTolerance,
        expectedCenter + centerTolerance,
      );
      widthAtBand = clamp(
        measuredWidth,
        Math.max(0.02, expectedWidth * 0.84),
        expectedWidth * 1.18,
      );
    } else if (leftObserved) {
      const measuredCenter = left.points[index].x + expectedWidth / 2;
      center = interpolate(expectedCenter, measuredCenter, 0.7);
    } else if (rightObserved) {
      const measuredCenter = right.points[index].x - expectedWidth / 2;
      center = interpolate(expectedCenter, measuredCenter, 0.7);
    }

    leftPoints.push({ x: center - widthAtBand / 2, y });
    rightPoints.push({ x: center + widthAtBand / 2, y });
  }

  return {
    left: { ...left, points: leftPoints },
    right: { ...right, points: rightPoints },
  };
}

function chooseLanePair(
  leftCandidates: FittedLine[],
  rightCandidates: FittedLine[],
  roadBottom: number,
  previous: DetectedLane | null,
) {
  let best:
    | {
        left: FittedLine;
        right: FittedLine;
        vanishX: number;
        vanishY: number;
        topY: number;
        score: number;
      }
    | null = null;

  for (const left of leftCandidates) {
    for (const right of rightCandidates) {
      const denominator = left.slope - right.slope;
      if (Math.abs(denominator) < 0.12) continue;
      const vanishY = (right.intercept - left.intercept) / denominator;
      const vanishX = left.slope * vanishY + left.intercept;
      if (
        vanishY < 0.16 ||
        vanishY > 0.6 ||
        vanishX < 0.2 ||
        vanishX > 0.8
      ) {
        continue;
      }

      const topY = clamp(
        vanishY + 0.055,
        0.35,
        Math.min(0.62, roadBottom - 0.22),
      );
      if (topY <= vanishY + 0.015) continue;
      const leftTop = left.slope * topY + left.intercept;
      const rightTop = right.slope * topY + right.intercept;
      const leftBottom = left.slope * roadBottom + left.intercept;
      const rightBottom = right.slope * roadBottom + right.intercept;
      const topWidth = rightTop - leftTop;
      const bottomWidth = rightBottom - leftBottom;
      const bottomCenter = (leftBottom + rightBottom) / 2;
      if (
        topWidth < 0.018 ||
        bottomWidth < 0.2 ||
        bottomWidth > 1.04 ||
        bottomCenter < 0.2 ||
        bottomCenter > 0.8 ||
        leftBottom >= 0.5 ||
        rightBottom <= 0.5
      ) {
        continue;
      }

      const leftDistance = 0.5 - leftBottom;
      const rightDistance = rightBottom - 0.5;
      let score =
        (left.confidence + right.confidence) * 0.75 +
        clamp((0.92 - bottomWidth) / 0.72, 0, 1) * 0.72 +
        clamp(1 - Math.abs(vanishX - 0.5) * 2.2, 0, 1) * 0.32 -
        Math.abs(leftDistance - rightDistance) * 0.35;
      if (previous) {
        const previousBounds = laneBoundsAt(previous, roadBottom);
        const previousWidth = previousBounds.right - previousBounds.left;
        const previousCenter =
          (previousBounds.left + previousBounds.right) / 2;
        score -= Math.abs(previousWidth - bottomWidth) * 1.25;
        score -= Math.abs(previousCenter - bottomCenter) * 1.55;
      }
      if (!best || score > best.score) {
        best = { left, right, vanishX, vanishY, topY, score };
      }
    }
  }
  return best;
}

function detectLane(
  pixels: Uint8ClampedArray,
  luminance: Float32Array,
  width: number,
  height: number,
  dashboard: DashboardMask | null,
  previous: DetectedLane | null,
): DetectedLane | null {
  const roiTop = 0.3;
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
  const pair = chooseLanePair(
    fitLaneLines(edgePoints, "left", roiTop, roadBottom),
    fitLaneLines(edgePoints, "right", roiTop, roadBottom),
    roadBottom,
    previous,
  );
  if (!pair) return null;
  const { left, right, vanishX, topY } = pair;
  const leftTrace = traceBoundaryFromImage(
    edgePoints,
    left,
    "left",
    topY,
    roadBottom,
    previous?.left ?? null,
  );
  const rightTrace = traceBoundaryFromImage(
    edgePoints,
    right,
    "right",
    topY,
    roadBottom,
    previous?.right ?? null,
  );
  if (
    leftTrace.observedRatio < 0.38 ||
    rightTrace.observedRatio < 0.38
  ) {
    return null;
  }

  const regularized = regularizeLanePair(
    leftTrace,
    rightTrace,
    left,
    right,
    previous,
  );
  const stableLeftTrace = regularized.left;
  const stableRightTrace = regularized.right;

  const leftTop = stableLeftTrace.points[0].x;
  const rightTop = stableRightTrace.points[0].x;
  const leftBottom =
    stableLeftTrace.points[stableLeftTrace.points.length - 1].x;
  const rightBottom =
    stableRightTrace.points[stableRightTrace.points.length - 1].x;
  const topWidth = rightTop - leftTop;
  const bottomWidth = rightBottom - leftBottom;
  const bottomCenter = (leftBottom + rightBottom) / 2;

  if (
    topWidth < 0.018 ||
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
  if (perspectiveGrowth < 1.18 || geometryConfidence < 0.38) return null;

  const widths = stableLeftTrace.points.map(
    (point, index) => stableRightTrace.points[index].x - point.x,
  );
  if (widths.some((widthAtBand) => widthAtBand <= 0.018)) return null;
  let severeNarrowing = 0;
  for (let index = 1; index < widths.length; index += 1) {
    if (widths[index] + 0.035 < widths[index - 1]) severeNarrowing += 1;
  }
  if (severeNarrowing > 2) return null;

  const confidence = clamp(
    Math.min(stableLeftTrace.confidence, stableRightTrace.confidence) *
      0.72 +
      geometryConfidence * 0.28,
    0,
    1,
  );
  if (confidence < 0.34) return null;
  return {
    left: { points: stableLeftTrace.points },
    right: { points: stableRightTrace.points },
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
    tracker.laneMisses += 1;
    tracker.laneEvidence = Math.max(0, tracker.laneEvidence - 1);
    if (tracker.lane && tracker.laneMisses <= 3) {
      tracker.lane = {
        ...tracker.lane,
        confidence: tracker.lane.confidence * 0.92,
      };
      return tracker.lane;
    }
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
    difference <= 0.085 &&
    Math.abs(previous.bottomY - raw.bottomY) <= 0.075;
  if (
    previous &&
    !consistent &&
    tracker.laneEvidence >= 2 &&
    tracker.laneMisses < 3 &&
    raw.confidence < previous.confidence + 0.18
  ) {
    tracker.laneMisses += 1;
    tracker.lane = {
      ...previous,
      confidence: previous.confidence * 0.94,
    };
    return tracker.lane;
  }

  tracker.laneMisses = 0;
  tracker.laneEvidence = consistent
    ? Math.min(6, tracker.laneEvidence + 1)
    : 1;
  tracker.lane = consistent
    ? {
        left: {
          points: blendPoints(raw.left.points, previous.left.points, 0.68),
        },
        right: {
          points: blendPoints(raw.right.points, previous.right.points, 0.68),
        },
        topY: interpolate(previous.topY, raw.topY, 0.68),
        bottomY: interpolate(previous.bottomY, raw.bottomY, 0.68),
        confidence: interpolate(previous.confidence, raw.confidence, 0.58),
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
    tracker.laneMisses = 0;
    tracker.dashboardEvidence = 0;
    return emptyRoadScene(analyzedAt);
  }

  const luminance = luminanceBuffer(
    pixels,
    width,
    height,
    tracker.scratchLuminance,
  );
  tracker.scratchLuminance = luminance;
  const rawDashboard = detectDashboard(luminance, width, height);
  const dashboard = stabilizeDashboard(rawDashboard, tracker);
  const rawLane = detectLane(
    pixels,
    luminance,
    width,
    height,
    rawDashboard ?? dashboard,
    tracker.lane,
  );
  const lane = stabilizeLane(rawLane, tracker);
  return { lane, dashboard, analyzedAt };
}

export function analyzeRoadSceneScaled(
  pixels: Uint8ClampedArray,
  sourceWidth: number,
  sourceHeight: number,
  tracker: RoadSceneTracker,
  analyzedAt: number,
  maxDimension = 256,
): RoadScene {
  const scale = Math.min(
    1,
    maxDimension / Math.max(sourceWidth, sourceHeight),
  );
  const width = Math.max(96, Math.round(sourceWidth * scale));
  const height = Math.max(96, Math.round(sourceHeight * scale));
  if (width === sourceWidth && height === sourceHeight) {
    return analyzeRoadScene(
      pixels,
      sourceWidth,
      sourceHeight,
      tracker,
      analyzedAt,
    );
  }

  const requiredLength = width * height * 4;
  const scaled =
    tracker.scratchPixels?.length === requiredLength
      ? tracker.scratchPixels
      : new Uint8ClampedArray(requiredLength);
  tracker.scratchPixels = scaled;

  for (let y = 0; y < height; y += 1) {
    const sourceY = Math.min(
      sourceHeight - 1,
      Math.floor((y * sourceHeight) / height),
    );
    for (let x = 0; x < width; x += 1) {
      const sourceX = Math.min(
        sourceWidth - 1,
        Math.floor((x * sourceWidth) / width),
      );
      const sourceIndex = (sourceY * sourceWidth + sourceX) * 4;
      const targetIndex = (y * width + x) * 4;
      scaled[targetIndex] = pixels[sourceIndex];
      scaled[targetIndex + 1] = pixels[sourceIndex + 1];
      scaled[targetIndex + 2] = pixels[sourceIndex + 2];
      scaled[targetIndex + 3] = pixels[sourceIndex + 3];
    }
  }

  return analyzeRoadScene(
    scaled,
    width,
    height,
    tracker,
    analyzedAt,
  );
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
