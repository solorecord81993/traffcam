import type { RoadScene } from "./road-scene";
import type {
  AnalyzedDetection,
  TravelMode,
  VisionAlert,
} from "./vision";

export type FrameRateTarget = 30 | 45 | 60;

export type CameraTelemetry = {
  type: "telemetry";
  sentAt: number;
  mode: TravelMode;
  frameRateTarget: FrameRateTarget;
  soundEnabled: boolean;
  detectedCount: number;
  speedKmh: number;
  gpsLabel: string;
  cameraFps: number;
  displayFps: number;
  inferenceFps: number;
  viewers: number;
  frameWidth: number;
  frameHeight: number;
  detections: AnalyzedDetection[];
  alert: VisionAlert | null;
  roadScene: RoadScene;
};

export type CameraControl =
  | { type: "control"; action: "mode"; value: TravelMode }
  | { type: "control"; action: "frameRate"; value: FrameRateTarget }
  | { type: "control"; action: "sound"; value: boolean };

export type RoomMessage =
  | CameraTelemetry
  | CameraControl
  | { type: "hello"; protocol: 1 };

const ROOM_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";
export const ROOM_CODE_LENGTH = 6;

export function createRoomCode() {
  const values = new Uint8Array(ROOM_CODE_LENGTH);
  crypto.getRandomValues(values);
  return Array.from(
    values,
    (value) => ROOM_ALPHABET[value % ROOM_ALPHABET.length],
  ).join("");
}

export function cleanRoomCode(value: string) {
  return value
    .toUpperCase()
    .replace(/[^2-9A-HJ-NP-Z]/g, "")
    .slice(0, ROOM_CODE_LENGTH);
}

export function roomPeerId(code: string) {
  return `roadguard-${cleanRoomCode(code).toLowerCase()}`;
}

export function isCameraTelemetry(value: unknown): value is CameraTelemetry {
  return Boolean(
    value &&
      typeof value === "object" &&
      "type" in value &&
      value.type === "telemetry",
  );
}

export function isCameraControl(value: unknown): value is CameraControl {
  if (!value || typeof value !== "object" || !("type" in value)) {
    return false;
  }
  if (value.type !== "control" || !("action" in value) || !("value" in value)) {
    return false;
  }
  if (value.action === "mode") {
    return ["walk", "ride", "drive"].includes(String(value.value));
  }
  if (value.action === "frameRate") {
    return [30, 45, 60].includes(Number(value.value));
  }
  return value.action === "sound" && typeof value.value === "boolean";
}
