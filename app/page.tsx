"use client";

import {
  AlertTriangle,
  Bike,
  Camera,
  CarFront,
  Footprints,
  Info,
  LoaderCircle,
  Maximize2,
  Navigation,
  RefreshCw,
  ShieldCheck,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  deriveMotion,
  EMPTY_MOTION,
  formatSpeed,
  freshMotion,
  type GeoFix,
} from "./motion";
import {
  analyzeRoadSceneScaled,
  createRoadSceneTracker,
  emptyRoadScene,
  type RoadScene,
} from "./road-scene";
import {
  analyzeDetections,
  formatDistance,
  renderVisionOverlay,
  selectAlert,
  type AnalyzedDetection,
  type TrackStore,
  type TravelMode,
  type VisionAlert,
} from "./vision";

type CameraState =
  | "idle"
  | "requesting"
  | "resuming"
  | "paused"
  | "running"
  | "error";
type ModelState = "idle" | "loading" | "ready" | "error";
type GpsState =
  | "idle"
  | "locating"
  | "ready"
  | "weak"
  | "denied"
  | "unsupported";

type RuntimeProfile = {
  ios: boolean;
  recovery: boolean;
  cameraMaxWidth: number;
  cameraMaxFps: number;
  inputMax: number;
  overlayMax: number;
  roadMax: number;
  roadIntervalMoving: number;
  roadIntervalStopped: number;
  overlayInterval: number;
  inferenceMovingCooldown: number;
  inferenceStoppedCooldown: number;
};

type StoredVisionSession = {
  at: number;
  mode: TravelMode;
  deviceId?: string;
  hidden: boolean;
};

type StartCameraOptions = {
  resume?: boolean;
  forceRecovery?: boolean;
};

type StopCameraOptions = {
  clearSession?: boolean;
  resetUi?: boolean;
};

type WorkerBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
  cls: number;
  name: "obstacle";
};

type WorkerResult = {
  type: "result";
  requestId: number;
  boxes: WorkerBox[];
  inferenceMs: number;
};

type WorkerResolver = {
  resolve: (result: WorkerResult) => void;
  reject: (error: Error) => void;
};

const VISION_SESSION_KEY = "roadguard-vision-session-v3";
const VISION_SESSION_MAX_AGE = 4 * 60 * 60 * 1000;
const CAMERA_REQUEST_TIMEOUT = 15000;
const DEFAULT_RUNTIME_PROFILE: RuntimeProfile = {
  ios: false,
  recovery: false,
  cameraMaxWidth: 1280,
  cameraMaxFps: 30,
  inputMax: 360,
  overlayMax: 1920,
  roadMax: 272,
  roadIntervalMoving: 420,
  roadIntervalStopped: 700,
  overlayInterval: 33,
  inferenceMovingCooldown: 80,
  inferenceStoppedCooldown: 180,
};

const MODES = {
  walk: {
    label: "เดิน",
    icon: Footprints,
    accent: "#74f7c5",
    copy: "เตือนสิ่งกีดขวางและยานพาหนะในระยะใกล้",
  },
  ride: {
    label: "ขี่",
    icon: Bike,
    accent: "#7ee7ff",
    copy: "สำหรับจักรยานและมอเตอร์ไซค์ที่ติดตั้งมือถือมั่นคง",
  },
  drive: {
    label: "ขับรถ",
    icon: CarFront,
    accent: "#ffd166",
    copy: "มองรถ คน และสิ่งกีดขวางในแนวทางข้างหน้า",
  },
} satisfies Record<
  TravelMode,
  {
    label: string;
    icon: typeof Footprints;
    accent: string;
    copy: string;
  }
>;

function ModeIcon({ mode, size = 21 }: { mode: TravelMode; size?: number }) {
  const Icon = MODES[mode].icon;
  return <Icon aria-hidden="true" size={size} strokeWidth={2.2} />;
}

function captureOrientation() {
  return window.innerHeight >= window.innerWidth
    ? ("portrait" as const)
    : ("landscape" as const);
}

function cameraConstraints(
  profile: RuntimeProfile,
  deviceId?: string,
): MediaTrackConstraints {
  return {
    width: {
      ideal: profile.cameraMaxWidth,
      max: profile.cameraMaxWidth,
    },
    frameRate: {
      ideal: profile.cameraMaxFps,
      max: profile.cameraMaxFps,
    },
    ...(deviceId
      ? {
          deviceId: { exact: deviceId },
        }
      : { facingMode: { ideal: "environment" } }),
  };
}

function cameraPriority(device: MediaDeviceInfo) {
  const label = device.label.toLowerCase();
  if (/(front|user|face|facetime|selfie|true.?depth)/i.test(label)) {
    return 0;
  }
  if (
    /(back|rear|environment|main|ultra|wide|tele|0\.5x|1x|2x|3x|5x)/i.test(
      label,
    )
  ) {
    return 2;
  }
  return 1;
}

function sortCameras(devices: MediaDeviceInfo[]) {
  return devices
    .map((device, index) => ({
      device,
      index,
      priority: cameraPriority(device),
    }))
    .sort((left, right) => right.priority - left.priority || left.index - right.index)
    .map(({ device }) => device);
}

function isIOSDevice() {
  return (
    /iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1)
  );
}

function browserStorage(
  kind: "localStorage" | "sessionStorage",
): Storage | null {
  try {
    return window[kind];
  } catch {
    return null;
  }
}

function readVisionSession(storage: Storage | null) {
  if (!storage) return null;
  try {
    const raw = storage.getItem(VISION_SESSION_KEY);
    if (!raw) return null;
    const stored = JSON.parse(raw) as Partial<StoredVisionSession>;
    if (
      typeof stored.at !== "number" ||
      Date.now() - stored.at >= VISION_SESSION_MAX_AGE ||
      !["walk", "ride", "drive"].includes(stored.mode ?? "")
    ) {
      storage.removeItem(VISION_SESSION_KEY);
      return null;
    }
    return {
      at: stored.at,
      mode: stored.mode as TravelMode,
      deviceId:
        typeof stored.deviceId === "string"
          ? stored.deviceId
          : undefined,
      hidden: Boolean(stored.hidden),
    } satisfies StoredVisionSession;
  } catch {
    return null;
  }
}

function hadUncleanVisionSession() {
  const stored = readVisionSession(browserStorage("localStorage"));
  return Boolean(stored && !stored.hidden);
}

function readResumableVisionSession() {
  return (
    readVisionSession(browserStorage("sessionStorage")) ??
    readVisionSession(browserStorage("localStorage"))
  );
}

function saveVisionSession(
  mode: TravelMode,
  deviceId?: string,
  hidden = false,
) {
  const serialized = JSON.stringify({
    at: Date.now(),
    mode,
    deviceId,
    hidden,
  } satisfies StoredVisionSession);
  for (const storage of [
    browserStorage("localStorage"),
    browserStorage("sessionStorage"),
  ]) {
    if (!storage) continue;
    try {
      storage.setItem(VISION_SESSION_KEY, serialized);
    } catch {
      // Private browsing or storage policies can disable a storage area.
    }
  }
}

function clearVisionSession() {
  for (const storage of [
    browserStorage("localStorage"),
    browserStorage("sessionStorage"),
  ]) {
    if (!storage) continue;
    try {
      storage.removeItem(VISION_SESSION_KEY);
    } catch {
      // Private browsing or storage policies can disable a storage area.
    }
  }
}

function selectRuntimeProfile(recovery: boolean): RuntimeProfile {
  if (!isIOSDevice()) return DEFAULT_RUNTIME_PROFILE;
  if (recovery) {
    return {
      ios: true,
      recovery: true,
      cameraMaxWidth: 720,
      cameraMaxFps: 24,
      inputMax: 288,
      overlayMax: 960,
      roadMax: 208,
      roadIntervalMoving: 140,
      roadIntervalStopped: 240,
      overlayInterval: 33,
      inferenceMovingCooldown: 180,
      inferenceStoppedCooldown: 300,
    };
  }
  return {
    ios: true,
    recovery: false,
    cameraMaxWidth: 1280,
    cameraMaxFps: 30,
    inputMax: 360,
    overlayMax: 1280,
    roadMax: 256,
    roadIntervalMoving: 100,
    roadIntervalStopped: 180,
    overlayInterval: 33,
    inferenceMovingCooldown: 80,
    inferenceStoppedCooldown: 180,
  };
}

async function requestCameraStream(
  profile: RuntimeProfile,
  deviceId?: string,
) {
  try {
    return await getUserMediaWithTimeout({
      audio: false,
      video: cameraConstraints(profile, deviceId),
    });
  } catch (error) {
    const name =
      error && typeof error === "object" && "name" in error
        ? String(error.name)
        : "";
    const canRetryWithoutStoredDevice =
      Boolean(deviceId) &&
      [
        "AbortError",
        "ConstraintNotSatisfiedError",
        "DevicesNotFoundError",
        "NotFoundError",
        "NotReadableError",
        "OverconstrainedError",
        "TrackStartError",
      ].includes(name);
    if (!canRetryWithoutStoredDevice) throw error;
    return getUserMediaWithTimeout({
      audio: false,
      video: cameraConstraints(profile),
    });
  }
}

function getUserMediaWithTimeout(
  constraints: MediaStreamConstraints,
) {
  return new Promise<MediaStream>((resolve, reject) => {
    let settled = false;
    const timer = window.setTimeout(() => {
      settled = true;
      reject(
        new DOMException(
          "Safari did not answer the camera request",
          "TimeoutError",
        ),
      );
    }, CAMERA_REQUEST_TIMEOUT);

    void navigator.mediaDevices.getUserMedia(constraints).then(
      (stream) => {
        if (settled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        settled = true;
        window.clearTimeout(timer);
        resolve(stream);
      },
      (error) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        reject(error);
      },
    );
  });
}

function visionFrameSize(
  sourceWidth: number,
  sourceHeight: number,
  maxDimension: number,
) {
  const scale = Math.min(
    1,
    maxDimension / Math.max(sourceWidth, sourceHeight),
  );
  return {
    width: Math.max(1, Math.round(sourceWidth * scale)),
    height: Math.max(1, Math.round(sourceHeight * scale)),
  };
}

async function resetCameraZoom(track: MediaStreamTrack) {
  try {
    const capabilities = track.getCapabilities() as MediaTrackCapabilities & {
      zoom?: { min: number; max: number; step?: number };
    };
    if (!capabilities.zoom) return;
    const zoom = Math.min(
      capabilities.zoom.max,
      Math.max(capabilities.zoom.min, 1),
    );
    const zoomConstraint = { zoom } as unknown as MediaTrackConstraintSet;
    await track.applyConstraints({ advanced: [zoomConstraint] });
  } catch {
    // Some iOS camera drivers advertise zoom but reject web constraints.
  }
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inferenceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const roadCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const geolocationWatchRef = useRef<number | null>(null);
  const visionHeartbeatRef = useRef<ReturnType<typeof setInterval> | null>(
    null,
  );
  const wakeLockRef = useRef<{ release: () => Promise<void> } | null>(null);
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backgroundReleaseTimerRef = useRef<ReturnType<
    typeof setTimeout
  > | null>(null);
  const obstacleWorkerRef = useRef<Worker | null>(null);
  const obstacleWorkerPromiseRef = useRef<Promise<void> | null>(null);
  const obstacleModelReadyRef = useRef(false);
  const obstacleRequestIdRef = useRef(0);
  const obstacleResolversRef = useRef(new Map<number, WorkerResolver>());
  const runningRef = useRef(false);
  const activeDeviceIdRef = useRef<string | undefined>(undefined);
  const resumeInFlightRef = useRef(false);
  const autoResumeAttemptedRef = useRef(false);
  const cameraSessionRef = useRef(0);
  const modeRef = useRef<TravelMode>("drive");
  const soundEnabledRef = useRef(true);
  const inferenceBusyRef = useRef(false);
  const inferenceErrorCountRef = useRef(0);
  const lastInferenceAtRef = useRef(0);
  const lastOverlayAtRef = useRef(0);
  const lastRoadAnalysisAtRef = useRef(0);
  const inferenceIntervalRef = useRef(260);
  const detectionsRef = useRef<AnalyzedDetection[]>([]);
  const roadSceneRef = useRef<RoadScene>(emptyRoadScene());
  const roadSceneTrackerRef = useRef(createRoadSceneTracker());
  const frameSizeRef = useRef({ width: 0, height: 0 });
  const overlaySizeRef = useRef({ width: 0, height: 0 });
  const captureOrientationRef = useRef<
    "portrait" | "landscape" | null
  >(null);
  const orientationTimerRef = useRef<ReturnType<
    typeof setTimeout
  > | null>(null);
  const alertRef = useRef<VisionAlert | null>(null);
  const alertHoldRef = useRef<{
    alert: VisionAlert | null;
    lastSeenAt: number;
  }>({ alert: null, lastSeenAt: 0 });
  const lastGeoFixRef = useRef<GeoFix | null>(null);
  const motionRef = useRef({ ...EMPTY_MOTION });
  const trackStoreRef = useRef<TrackStore>({
    nextId: 1,
    tracks: new Map(),
  });
  const audioContextRef = useRef<AudioContext | null>(null);
  const lastAudioAlertRef = useRef({ key: "", level: "", at: 0 });
  const statsRef = useRef({ startedAt: 0, frames: 0 });
  const runtimeProfileRef = useRef<RuntimeProfile>(
    DEFAULT_RUNTIME_PROFILE,
  );
  const recoveryCheckedRef = useRef<boolean | null>(null);

  const [mode, setMode] = useState<TravelMode>("drive");
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [modelProgress, setModelProgress] = useState(0);
  const [modelBackend, setModelBackend] = useState("");
  const [runtimeLabel, setRuntimeLabel] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [modelError, setModelError] = useState("");
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [controlsVisible, setControlsVisible] = useState(true);
  const [showInfo, setShowInfo] = useState(false);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [cameraIndex, setCameraIndex] = useState(0);
  const [fps, setFps] = useState(0);
  const [detectedCount, setDetectedCount] = useState(0);
  const [alert, setAlert] = useState<VisionAlert | null>(null);
  const [gpsState, setGpsState] = useState<GpsState>("idle");
  const [speedKmh, setSpeedKmh] = useState(0);
  const [sessionChecked, setSessionChecked] = useState(false);

  const currentMode = MODES[mode];
  const isRunning =
    cameraState === "running" ||
    cameraState === "resuming" ||
    cameraState === "paused";

  const modeEntries = useMemo(
    () => Object.entries(MODES) as [TravelMode, (typeof MODES)[TravelMode]][],
    [],
  );

  const changeMode = useCallback((nextMode: TravelMode) => {
    if (modeRef.current === nextMode) return;
    modeRef.current = nextMode;
    trackStoreRef.current = { nextId: 1, tracks: new Map() };
    roadSceneRef.current = emptyRoadScene();
    roadSceneTrackerRef.current = createRoadSceneTracker();
    detectionsRef.current = [];
    alertRef.current = null;
    alertHoldRef.current = { alert: null, lastSeenAt: 0 };
    setAlert(null);
    setDetectedCount(0);
    setControlsVisible(true);
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
    if (runningRef.current) {
      saveVisionSession(
        nextMode,
        activeDeviceIdRef.current,
        document.hidden,
      );
      hideTimerRef.current = setTimeout(
        () => setControlsVisible(false),
        5000,
      );
    }
    setMode(nextMode);
  }, []);

  useEffect(() => {
    soundEnabledRef.current = soundEnabled;
  }, [soundEnabled]);

  const unlockAudio = useCallback(async () => {
    try {
      const AudioContextConstructor =
        window.AudioContext ??
        (
          window as typeof window & {
            webkitAudioContext?: typeof AudioContext;
          }
        ).webkitAudioContext;
      if (!AudioContextConstructor) return;
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContextConstructor();
      }
      if (audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume();
      }
      window.speechSynthesis?.getVoices();
    } catch {
      audioContextRef.current = null;
    }
  }, []);

  const playWarning = useCallback(
    (nextAlert: VisionAlert) => {
      if (!soundEnabledRef.current) return;
      const now = Date.now();
      const previous = lastAudioAlertRef.current;
      const cooldown = nextAlert.level === "danger" ? 6500 : 14000;
      const escalated =
        nextAlert.level === "danger" && previous.level !== "danger";
      if (
        !escalated &&
        now - previous.at < cooldown &&
        previous.key === nextAlert.key
      ) {
        return;
      }
      if (!escalated && now - previous.at < 4200) return;
      lastAudioAlertRef.current = {
        key: nextAlert.key,
        level: nextAlert.level,
        at: now,
      };

      const context = audioContextRef.current;
      if (context && context.state === "running") {
        const danger = nextAlert.level === "danger";
        const pulses = danger ? [0, 0.19, 0.38] : [0];
        pulses.forEach((offset) => {
          const oscillator = context.createOscillator();
          const gain = context.createGain();
          oscillator.type = danger ? "square" : "sine";
          oscillator.frequency.setValueAtTime(
            danger ? 940 : 720,
            context.currentTime + offset,
          );
          gain.gain.setValueAtTime(0.0001, context.currentTime + offset);
          gain.gain.exponentialRampToValueAtTime(
            danger ? 0.22 : 0.13,
            context.currentTime + offset + 0.015,
          );
          gain.gain.exponentialRampToValueAtTime(
            0.0001,
            context.currentTime + offset + 0.13,
          );
          oscillator.connect(gain);
          gain.connect(context.destination);
          oscillator.start(context.currentTime + offset);
          oscillator.stop(context.currentTime + offset + 0.15);
        });
      }

      try {
        if ("speechSynthesis" in window) {
          window.speechSynthesis.cancel();
          const phrase =
            nextAlert.level === "danger"
              ? `อันตราย ${nextAlert.objectName}${nextAlert.direction}`
              : `ระวัง ${nextAlert.objectName}${nextAlert.direction}`;
          const utterance = new SpeechSynthesisUtterance(phrase);
          utterance.lang = "th-TH";
          utterance.rate = 1.05;
          utterance.pitch = 1;
          utterance.volume = 0.86;
          const thaiVoice = window.speechSynthesis
            .getVoices()
            .find((voice) => voice.lang.toLowerCase().startsWith("th"));
          if (thaiVoice) utterance.voice = thaiVoice;
          window.speechSynthesis.speak(utterance);
        }
        if (nextAlert.level === "danger") {
          navigator.vibrate?.([120, 70, 120]);
        }
      } catch {
        // Beep and on-screen warning remain available when speech is unavailable.
      }
    },
    [],
  );

  const loadObstacleModel = useCallback(async () => {
    if (obstacleWorkerRef.current) return;
    if (obstacleWorkerPromiseRef.current) {
      return obstacleWorkerPromiseRef.current;
    }

    const task = new Promise<void>((resolve, reject) => {
      obstacleModelReadyRef.current = false;
      setModelState("loading");
      setModelProgress(12);
      setModelError("");
      const worker = new Worker(
        new URL("./obstacle-worker.ts", import.meta.url),
        { type: "module" },
      );
      obstacleWorkerRef.current = worker;
      worker.onmessage = (event: MessageEvent) => {
        const message = event.data as
          | { type: "ready"; backend: string }
          | WorkerResult
          | {
              type: "error";
              stage: "load" | "infer";
              requestId?: number;
              message: string;
            };
        if (message.type === "ready") {
          obstacleModelReadyRef.current = true;
          setModelBackend(message.backend);
          setModelProgress(100);
          setModelState("ready");
          resolve();
          return;
        }
        if (message.type === "result") {
          const resolver = obstacleResolversRef.current.get(
            message.requestId,
          );
          if (!resolver) return;
          obstacleResolversRef.current.delete(message.requestId);
          resolver.resolve(message);
          return;
        }
        const error = new Error(message.message);
        if (message.requestId !== undefined) {
          const resolver = obstacleResolversRef.current.get(
            message.requestId,
          );
          if (resolver) {
            obstacleResolversRef.current.delete(message.requestId);
            resolver.reject(error);
          }
        }
        if (message.stage === "load") {
          worker.terminate();
          obstacleWorkerRef.current = null;
          obstacleWorkerPromiseRef.current = null;
          obstacleModelReadyRef.current = false;
          setModelError(message.message);
          setModelState("error");
          reject(error);
        }
      };
      worker.onerror = (event) => {
        const error = new Error(event.message || "AI worker หยุดทำงาน");
        for (const resolver of obstacleResolversRef.current.values()) {
          resolver.reject(error);
        }
        obstacleResolversRef.current.clear();
        worker.terminate();
        obstacleWorkerRef.current = null;
        obstacleWorkerPromiseRef.current = null;
        obstacleModelReadyRef.current = false;
        setModelError(error.message);
        setModelState("error");
        reject(error);
      };
      setModelProgress(28);
      worker.postMessage({
        type: "init",
        modelUrl: "/models/yolo26n.onnx",
      });
    });
    obstacleWorkerPromiseRef.current = task;
    return task;
  }, []);

  const releaseObstacleModelForBackground = useCallback(() => {
    const worker = obstacleWorkerRef.current;
    if (!worker || inferenceBusyRef.current) return false;
    worker.postMessage({ type: "dispose" });
    worker.terminate();
    obstacleWorkerRef.current = null;
    obstacleWorkerPromiseRef.current = null;
    obstacleModelReadyRef.current = false;
    setModelBackend("");
    setModelProgress(0);
    setModelState("idle");
    return true;
  }, []);

  const scheduleObstacleModelRelease = useCallback(() => {
    if (!isIOSDevice()) return;
    if (backgroundReleaseTimerRef.current) {
      clearTimeout(backgroundReleaseTimerRef.current);
      backgroundReleaseTimerRef.current = null;
    }
    let attempts = 0;
    const releaseWhenIdle = () => {
      backgroundReleaseTimerRef.current = null;
      if (!document.hidden || releaseObstacleModelForBackground()) return;
      attempts += 1;
      if (attempts >= 8) return;
      backgroundReleaseTimerRef.current = setTimeout(
        releaseWhenIdle,
        250,
      );
    };
    releaseWhenIdle();
  }, [releaseObstacleModelForBackground]);

  const requestWakeLock = useCallback(async () => {
    try {
      const wakeLockApi = (
        navigator as Navigator & {
          wakeLock?: {
            request: (type: "screen") => Promise<{
              release: () => Promise<void>;
            }>;
          };
        }
      ).wakeLock;
      wakeLockRef.current = wakeLockApi
        ? await wakeLockApi.request("screen")
        : null;
    } catch {
      wakeLockRef.current = null;
    }
  }, []);

  const stopLocationTracking = useCallback(() => {
    if (
      geolocationWatchRef.current !== null &&
      "geolocation" in navigator
    ) {
      navigator.geolocation.clearWatch(geolocationWatchRef.current);
    }
    geolocationWatchRef.current = null;
    lastGeoFixRef.current = null;
    motionRef.current = { ...EMPTY_MOTION };
    setSpeedKmh(0);
    setGpsState("idle");
  }, []);

  const startLocationTracking = useCallback(() => {
    if (!("geolocation" in navigator)) {
      setGpsState("unsupported");
      motionRef.current = { ...EMPTY_MOTION };
      return;
    }

    if (geolocationWatchRef.current !== null) {
      navigator.geolocation.clearWatch(geolocationWatchRef.current);
    }

    setGpsState("locating");
    geolocationWatchRef.current = navigator.geolocation.watchPosition(
      (position) => {
        const fix: GeoFix = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
          speedMps:
            position.coords.speed !== null &&
            Number.isFinite(position.coords.speed)
              ? position.coords.speed
              : null,
          timestamp: Number.isFinite(position.timestamp)
            ? position.timestamp
            : Date.now(),
        };
        const motion = deriveMotion(
          lastGeoFixRef.current,
          fix,
          motionRef.current,
        );
        lastGeoFixRef.current = fix;
        motionRef.current = motion;
        setSpeedKmh(formatSpeed(motion.speedMps));
        setGpsState(motion.reliable ? "ready" : "weak");
      },
      (error) => {
        motionRef.current = {
          ...motionRef.current,
          reliable: false,
        };
        if (error.code === error.PERMISSION_DENIED) {
          setGpsState("denied");
          setSpeedKmh(0);
        } else {
          setGpsState("weak");
        }
      },
      {
        enableHighAccuracy: true,
        maximumAge: 1000,
        timeout: 12000,
      },
    );
  }, []);

  const stopCamera = useCallback((options: StopCameraOptions = {}) => {
    const {
      clearSession = true,
      resetUi = true,
    } = options;
    runningRef.current = false;
    cameraSessionRef.current += 1;
    if (visionHeartbeatRef.current) {
      clearInterval(visionHeartbeatRef.current);
      visionHeartbeatRef.current = null;
    }
    if (clearSession) clearVisionSession();
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (clearSession) activeDeviceIdRef.current = undefined;
    stopLocationTracking();
    void wakeLockRef.current?.release();
    wakeLockRef.current = null;
    detectionsRef.current = [];
    roadSceneRef.current = emptyRoadScene();
    roadSceneTrackerRef.current = createRoadSceneTracker();
    frameSizeRef.current = { width: 0, height: 0 };
    overlaySizeRef.current = { width: 0, height: 0 };
    if (canvasRef.current) {
      canvasRef.current.width = 1;
      canvasRef.current.height = 1;
    }
    for (const offscreen of [
      inferenceCanvasRef.current,
      roadCanvasRef.current,
    ]) {
      if (!offscreen) continue;
      offscreen.width = 1;
      offscreen.height = 1;
    }
    inferenceErrorCountRef.current = 0;
    statsRef.current = { startedAt: 0, frames: 0 };
    alertRef.current = null;
    alertHoldRef.current = { alert: null, lastSeenAt: 0 };
    trackStoreRef.current = { nextId: 1, tracks: new Map() };
    setAlert(null);
    setDetectedCount(0);
    setFps(0);
    if (resetUi) setCameraState("idle");
  }, [stopLocationTracking]);

  const captureCurrentFrame = useCallback(
    (
      maxDimension: number,
      target: "inference" | "road",
    ) => {
      const video = videoRef.current;
      if (
        !video ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        !video.videoWidth ||
        !video.videoHeight
      ) {
        return null;
      }

      const dimensions = visionFrameSize(
        video.videoWidth,
        video.videoHeight,
        maxDimension,
      );
      const currentCanvas =
        target === "inference"
          ? inferenceCanvasRef.current
          : roadCanvasRef.current;
      let frameCanvas = currentCanvas;
      if (
        !frameCanvas ||
        frameCanvas.width !== dimensions.width ||
        frameCanvas.height !== dimensions.height
      ) {
        const nextCanvas = document.createElement("canvas");
        nextCanvas.width = dimensions.width;
        nextCanvas.height = dimensions.height;
        frameCanvas = nextCanvas;
        if (target === "inference") {
          inferenceCanvasRef.current = nextCanvas;
        } else {
          roadCanvasRef.current = nextCanvas;
        }
      }
      const context = frameCanvas.getContext("2d", {
        alpha: false,
        willReadFrequently: true,
      });
      if (!context) return null;
      context.drawImage(
        video,
        0,
        0,
        dimensions.width,
        dimensions.height,
      );
      return context.getImageData(
        0,
        0,
        dimensions.width,
        dimensions.height,
      );
    },
    [],
  );

  const runInference = useCallback(
    async (session: number, frame: ImageData) => {
      const worker = obstacleWorkerRef.current;
      if (
        !worker ||
        !obstacleModelReadyRef.current ||
        inferenceBusyRef.current ||
        document.hidden
      ) {
        return;
      }

      inferenceBusyRef.current = true;
      const startedAt = performance.now();
      const requestId = ++obstacleRequestIdRef.current;
      try {
        const results = await new Promise<WorkerResult>((resolve, reject) => {
          obstacleResolversRef.current.set(requestId, { resolve, reject });
          worker.postMessage(
            {
              type: "infer",
              requestId,
              width: frame.width,
              height: frame.height,
              pixels: frame.data.buffer,
            },
            [frame.data.buffer as ArrayBuffer],
          );
        });
        if (
          session !== cameraSessionRef.current ||
          !runningRef.current
        ) {
          return;
        }

        const motion = freshMotion(motionRef.current);
        if (!motion.reliable && motionRef.current.reliable) {
          motionRef.current = motion;
          setGpsState("weak");
        }
        const now = performance.now();
        const analyzed = analyzeDetections(
          results.boxes,
          frame.width,
          frame.height,
          modeRef.current,
          now,
          trackStoreRef.current,
          motion,
          roadSceneRef.current,
        );
        const nextAlert = selectAlert(analyzed, modeRef.current, motion);
        let visibleAlert = nextAlert;
        if (nextAlert) {
          alertHoldRef.current = { alert: nextAlert, lastSeenAt: now };
        } else if (
          alertHoldRef.current.alert &&
          now - alertHoldRef.current.lastSeenAt < 1400
        ) {
          visibleAlert = alertHoldRef.current.alert;
        } else {
          alertHoldRef.current = { alert: null, lastSeenAt: 0 };
        }
        detectionsRef.current = analyzed;
        alertRef.current = visibleAlert;
        setDetectedCount(analyzed.length);
        setAlert((previous) => {
          if (
            previous?.key === visibleAlert?.key &&
            previous?.level === visibleAlert?.level &&
            previous?.detail === visibleAlert?.detail
          ) {
            return previous;
          }
          return visibleAlert;
        });
        if (nextAlert) playWarning(nextAlert);

        inferenceErrorCountRef.current = 0;
        const elapsed = performance.now() - startedAt;
        const moving = !motion.reliable || motion.speedMps >= 1.2;
        const profile = runtimeProfileRef.current;
        const cooldown = moving
          ? profile.inferenceMovingCooldown
          : profile.inferenceStoppedCooldown;
        // lastInferenceAtRef records the start time, so include inference time
        // to guarantee a real cooldown instead of immediately starting again.
        inferenceIntervalRef.current = elapsed + cooldown;

        const stats = statsRef.current;
        if (!stats.startedAt) stats.startedAt = performance.now();
        stats.frames += 1;
        const statsElapsed = performance.now() - stats.startedAt;
        if (statsElapsed >= 1600) {
          setFps((stats.frames * 1000) / statsElapsed);
          statsRef.current = { startedAt: performance.now(), frames: 0 };
        }
      } catch (error) {
        console.error("Obstacle inference failed", error);
        inferenceErrorCountRef.current += 1;
        if (
          inferenceErrorCountRef.current >= 3 &&
          obstacleWorkerRef.current === worker
        ) {
          worker.terminate();
          obstacleWorkerRef.current = null;
          obstacleWorkerPromiseRef.current = null;
          obstacleModelReadyRef.current = false;
          detectionsRef.current = [];
          alertRef.current = null;
          alertHoldRef.current = { alert: null, lastSeenAt: 0 };
          setDetectedCount(0);
          setAlert(null);
          setModelBackend("");
          setModelError(
            "หยุด AI ชั่วคราวเพื่อป้องกัน Safari ปิดหน้า กด “ลองใหม่” เพื่อเริ่มใหม่",
          );
          setModelState("error");
        }
      } finally {
        inferenceBusyRef.current = false;
      }
    },
    [playWarning],
  );

  const beginVisionLoop = useCallback(
    (session: number) => {
      const frame = (elapsed: number) => {
        if (
          session !== cameraSessionRef.current ||
          !runningRef.current
        ) {
          return;
        }
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (video && canvas && video.videoWidth && video.videoHeight) {
          const profile = runtimeProfileRef.current;
          const analysisDimensions = visionFrameSize(
            video.videoWidth,
            video.videoHeight,
            profile.inputMax,
          );
          const overlayDimensions = visionFrameSize(
            video.videoWidth,
            video.videoHeight,
            profile.overlayMax,
          );
          const analysisDimensionsChanged =
            frameSizeRef.current.width !== analysisDimensions.width ||
            frameSizeRef.current.height !== analysisDimensions.height;
          const overlayDimensionsChanged =
            overlaySizeRef.current.width !== overlayDimensions.width ||
            overlaySizeRef.current.height !== overlayDimensions.height;
          if (overlayDimensionsChanged) {
            overlaySizeRef.current = overlayDimensions;
            canvas.width = overlayDimensions.width;
            canvas.height = overlayDimensions.height;
          }
          if (analysisDimensionsChanged) {
            frameSizeRef.current = analysisDimensions;
            trackStoreRef.current = { nextId: 1, tracks: new Map() };
            roadSceneRef.current = emptyRoadScene(elapsed);
            roadSceneTrackerRef.current = createRoadSceneTracker();
            detectionsRef.current = [];
            alertRef.current = null;
            alertHoldRef.current = { alert: null, lastSeenAt: 0 };
            setDetectedCount(0);
            setAlert(null);
          }

          const moving =
            !motionRef.current.reliable ||
            motionRef.current.speedMps >= 0.8;
          const roadInterval = moving
            ? profile.roadIntervalMoving
            : profile.roadIntervalStopped;
          const roadDue =
            modeRef.current === "drive" &&
            elapsed - lastRoadAnalysisAtRef.current >= roadInterval;
          const inferenceDue =
            obstacleModelReadyRef.current &&
            Boolean(obstacleWorkerRef.current) &&
            elapsed - lastInferenceAtRef.current >=
              inferenceIntervalRef.current;

          if (roadDue) {
            lastRoadAnalysisAtRef.current = elapsed;
            try {
              const roadFrame = captureCurrentFrame(
                profile.roadMax,
                "road",
              );
              if (roadFrame) {
                roadSceneRef.current = analyzeRoadSceneScaled(
                  roadFrame.data,
                  roadFrame.width,
                  roadFrame.height,
                  roadSceneTrackerRef.current,
                  elapsed,
                  profile.roadMax,
                );
              }
            } catch (error) {
              console.warn("Road frame capture failed", error);
              if (
                elapsed - roadSceneRef.current.analyzedAt >
                roadInterval * 3
              ) {
                roadSceneRef.current = emptyRoadScene(elapsed);
                roadSceneTrackerRef.current = createRoadSceneTracker();
              }
            }
          }

          if (inferenceDue && !inferenceBusyRef.current) {
            try {
              const inferenceFrame = captureCurrentFrame(
                profile.inputMax,
                "inference",
              );
              if (inferenceFrame) {
                lastInferenceAtRef.current = elapsed;
                void runInference(session, inferenceFrame);
              }
            } catch (error) {
              console.warn("Inference frame capture failed", error);
            }
          }

          if (
            elapsed - lastOverlayAtRef.current >=
            profile.overlayInterval
          ) {
            lastOverlayAtRef.current = elapsed;
            renderVisionOverlay(
              canvas,
              modeRef.current,
              elapsed,
              detectionsRef.current,
              alertRef.current,
              roadSceneRef.current,
              analysisDimensions.width,
              analysisDimensions.height,
            );
          }
        }
        animationRef.current = requestAnimationFrame(frame);
      };
      animationRef.current = requestAnimationFrame(frame);
    },
    [captureCurrentFrame, runInference],
  );

  const refreshCameraList = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput",
      );
      const uniqueVideoDevices = videoDevices.filter(
        (device, index, allDevices) =>
          !device.deviceId ||
          allDevices.findIndex(
            (candidate) => candidate.deviceId === device.deviceId,
          ) === index,
      );
      const orderedCameras = sortCameras(uniqueVideoDevices);
      setCameras(orderedCameras);
      const activeIndex = orderedCameras.findIndex(
        (device) => device.deviceId === activeDeviceIdRef.current,
      );
      setCameraIndex(activeIndex >= 0 ? activeIndex : 0);
    } catch {
      setCameras([]);
      setCameraIndex(0);
    }
  }, []);

  const startCamera = useCallback(
    async (
      deviceId?: string,
      options: StartCameraOptions = {},
    ) => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setErrorMessage("Safari รุ่นนี้ไม่รองรับการเปิดกล้องผ่านเว็บ");
        setCameraState("error");
        return;
      }

      await unlockAudio();
      const recovery = options.forceRecovery
        ? true
        : (recoveryCheckedRef.current ?? hadUncleanVisionSession());
      recoveryCheckedRef.current = recovery;
      const profile = selectRuntimeProfile(recovery);
      runtimeProfileRef.current = profile;
      setRuntimeLabel(
        profile.recovery
          ? "กู้คืน"
          : profile.ios
            ? "เสถียร"
            : "",
      );
      stopCamera({ clearSession: false, resetUi: false });
      const session = ++cameraSessionRef.current;
      lastInferenceAtRef.current = 0;
      lastOverlayAtRef.current = 0;
      lastRoadAnalysisAtRef.current = 0;
      inferenceIntervalRef.current = 0;
      inferenceErrorCountRef.current = 0;
      setCameraState(options.resume ? "resuming" : "requesting");
      setErrorMessage("");
      try {
        const orientation = captureOrientation();
        captureOrientationRef.current = orientation;
        const stream = await requestCameraStream(profile, deviceId);
        if (session !== cameraSessionRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        const cameraTrack = stream.getVideoTracks()[0];
        if (cameraTrack) await resetCameraZoom(cameraTrack);
        activeDeviceIdRef.current =
          cameraTrack?.getSettings().deviceId || deviceId;
        const video = videoRef.current;
        if (!video) throw new Error("camera-view-missing");
        video.srcObject = stream;
        await video.play();
        startLocationTracking();
        await refreshCameraList();
        await requestWakeLock();
        runningRef.current = true;
        void loadObstacleModel().catch(() => undefined);
        saveVisionSession(
          modeRef.current,
          activeDeviceIdRef.current,
          false,
        );
        visionHeartbeatRef.current = setInterval(
          () =>
            saveVisionSession(
              modeRef.current,
              activeDeviceIdRef.current,
              document.hidden,
            ),
          15000,
        );
        setCameraState("running");
        if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
        hideTimerRef.current = setTimeout(
          () => setControlsVisible(false),
          5000,
        );
        beginVisionLoop(session);
      } catch (error) {
        console.error(error);
        if (options.resume) {
          const errorName =
            error && typeof error === "object" && "name" in error
              ? String(error.name)
              : "";
          stopCamera({ clearSession: false, resetUi: false });
          saveVisionSession(
            modeRef.current,
            activeDeviceIdRef.current,
            false,
          );
          setErrorMessage(
            ["NotAllowedError", "PermissionDeniedError"].includes(
              errorName,
            )
              ? "สิทธิ์กล้องถูกปิด กรุณาอนุญาต Camera ให้เว็บไซต์นี้ใน Safari แล้วแตะอีกครั้ง"
              : errorName === "TimeoutError"
                ? "Safari ไม่ตอบคำขอกล้องภายใน 15 วินาที แตะเพื่อลองขอสิทธิ์ใหม่"
                : "Safari ต้องให้แตะหน้าจอหนึ่งครั้งเพื่อเปิดกล้องต่อ",
          );
          setCameraState("paused");
        } else {
          stopCamera({ clearSession: true, resetUi: false });
          setErrorMessage(
            "เปิดกล้องไม่ได้ กรุณาอนุญาต Camera ใน Safari แล้วลองอีกครั้ง",
          );
          setCameraState("error");
        }
      }
    },
    [
      beginVisionLoop,
      loadObstacleModel,
      refreshCameraList,
      requestWakeLock,
      startLocationTracking,
      stopCamera,
      unlockAudio,
    ],
  );

  const cycleCamera = useCallback(async () => {
    if (cameras.length < 2) return;
    const activeIndex = cameras.findIndex(
      (device) => device.deviceId === activeDeviceIdRef.current,
    );
    const currentIndex = activeIndex >= 0 ? activeIndex : cameraIndex;
    const nextIndex = (currentIndex + 1) % cameras.length;
    const nextCamera = cameras[nextIndex];
    if (!nextCamera?.deviceId) return;
    setCameraIndex(nextIndex);
    await startCamera(nextCamera.deviceId, { resume: true });
  }, [cameraIndex, cameras, startCamera]);

  useEffect(() => {
    const mediaDevices = navigator.mediaDevices;
    if (!mediaDevices?.addEventListener) return;
    const handleDeviceChange = () => {
      void refreshCameraList();
    };
    mediaDevices.addEventListener("devicechange", handleDeviceChange);
    return () => {
      mediaDevices.removeEventListener("devicechange", handleDeviceChange);
    };
  }, [refreshCameraList]);

  const resumeCamera = useCallback(async () => {
    if (resumeInFlightRef.current) {
      cameraSessionRef.current += 1;
      resumeInFlightRef.current = false;
    }
    resumeInFlightRef.current = true;
    try {
      await startCamera(activeDeviceIdRef.current, {
        resume: true,
        forceRecovery: true,
      });
    } finally {
      resumeInFlightRef.current = false;
    }
  }, [startCamera]);

  const cancelResume = useCallback(() => {
    recoveryCheckedRef.current = null;
    setErrorMessage("");
    stopCamera();
  }, [stopCamera]);

  const toggleSound = useCallback(async () => {
    const next = !soundEnabledRef.current;
    if (next) await unlockAudio();
    soundEnabledRef.current = next;
    setSoundEnabled(next);
    if (!next) window.speechSynthesis?.cancel();
  }, [unlockAudio]);

  const retryModel = useCallback(() => {
    const recoveryProfile = selectRuntimeProfile(true);
    runtimeProfileRef.current = recoveryProfile;
    setRuntimeLabel(recoveryProfile.ios ? "กู้คืน" : "");
    inferenceErrorCountRef.current = 0;
    inferenceIntervalRef.current = 0;
    obstacleWorkerRef.current?.terminate();
    obstacleWorkerRef.current = null;
    obstacleWorkerPromiseRef.current = null;
    obstacleModelReadyRef.current = false;
    void loadObstacleModel().catch(() => undefined);
  }, [loadObstacleModel]);

  const revealControls = useCallback(() => {
    setControlsVisible(true);
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
    if (isRunning && !showInfo) {
      hideTimerRef.current = setTimeout(() => setControlsVisible(false), 5000);
    }
  }, [isRunning, showInfo]);

  const enterFullscreen = useCallback(async () => {
    try {
      if (!document.fullscreenElement) {
        if (!document.documentElement.requestFullscreen) {
          setShowInfo(true);
          return;
        }
        await document.documentElement.requestFullscreen();
      } else {
        await document.exitFullscreen?.();
      }
    } catch {
      setShowInfo(true);
    }
  }, []);

  const restoreCameraAfterVisibility = useCallback(async () => {
    if (
      document.hidden ||
      !runningRef.current ||
      resumeInFlightRef.current
    ) {
      return;
    }

    saveVisionSession(
      modeRef.current,
      activeDeviceIdRef.current,
      false,
    );
    await requestWakeLock();

    const track = streamRef.current?.getVideoTracks()[0];
    const video = videoRef.current;
    if (track?.readyState === "live" && video) {
      try {
        if (video.srcObject !== streamRef.current) {
          video.srcObject = streamRef.current;
        }
        if (video.paused) await video.play();
        void loadObstacleModel().catch(() => undefined);
        return;
      } catch {
        // Restart the stream below if iOS did not restore playback.
      }
    }

    if (isIOSDevice()) {
      stopCamera({ clearSession: false, resetUi: false });
      saveVisionSession(
        modeRef.current,
        activeDeviceIdRef.current,
        false,
      );
      setErrorMessage(
        "แตะปุ่มด้านล่างเพื่อให้ Safari เปิดกล้องอีกครั้ง",
      );
      setCameraState("paused");
      return;
    }

    resumeInFlightRef.current = true;
    try {
      await startCamera(activeDeviceIdRef.current, {
        resume: true,
        forceRecovery: true,
      });
    } finally {
      resumeInFlightRef.current = false;
    }
  }, [loadObstacleModel, requestWakeLock, startCamera, stopCamera]);

  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.hidden) {
        if (runningRef.current) {
          saveVisionSession(
            modeRef.current,
            activeDeviceIdRef.current,
            true,
          );
        }
        scheduleObstacleModelRelease();
        return;
      }
      if (backgroundReleaseTimerRef.current) {
        clearTimeout(backgroundReleaseTimerRef.current);
        backgroundReleaseTimerRef.current = null;
      }
      void restoreCameraAfterVisibility();
    };
    const onPageHide = () => {
      if (runningRef.current) {
        saveVisionSession(
          modeRef.current,
          activeDeviceIdRef.current,
          true,
        );
      }
      scheduleObstacleModelRelease();
    };
    const onPageShow = () => {
      void restoreCameraAfterVisibility();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    window.addEventListener("pagehide", onPageHide);
    window.addEventListener("pageshow", onPageShow);
    return () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("pagehide", onPageHide);
      window.removeEventListener("pageshow", onPageShow);
    };
  }, [restoreCameraAfterVisibility, scheduleObstacleModelRelease]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const stored = readResumableVisionSession();
      if (!stored) {
        setSessionChecked(true);
        return;
      }
      if (autoResumeAttemptedRef.current || runningRef.current) {
        setSessionChecked(true);
        return;
      }
      autoResumeAttemptedRef.current = true;
      activeDeviceIdRef.current = stored.deviceId;
      recoveryCheckedRef.current = true;
      modeRef.current = stored.mode;
      setMode(stored.mode);
      setSessionChecked(true);
      if (isIOSDevice()) {
        setErrorMessage(
          "แตะปุ่มเพื่อเปิดกล้องและอนุญาตให้ Safari ทำงานต่อ",
        );
        setCameraState("paused");
        return;
      }
      setCameraState("resuming");
      resumeInFlightRef.current = true;
      void startCamera(stored.deviceId, {
        resume: true,
        forceRecovery: true,
      }).finally(() => {
        resumeInFlightRef.current = false;
      });
    }, 0);

    return () => window.clearTimeout(timer);
  }, [startCamera]);

  useEffect(() => {
    const applyOrientation = () => {
      const orientation = captureOrientation();
      if (captureOrientationRef.current === orientation) return;
      captureOrientationRef.current = orientation;
      roadSceneRef.current = emptyRoadScene(performance.now());
      roadSceneTrackerRef.current = createRoadSceneTracker();
      detectionsRef.current = [];
      trackStoreRef.current = { nextId: 1, tracks: new Map() };
      alertRef.current = null;
      alertHoldRef.current = { alert: null, lastSeenAt: 0 };
      setDetectedCount(0);
      setAlert(null);
    };
    const scheduleOrientation = () => {
      if (orientationTimerRef.current) {
        clearTimeout(orientationTimerRef.current);
      }
      orientationTimerRef.current = setTimeout(applyOrientation, 180);
    };

    window.addEventListener("resize", scheduleOrientation);
    window.screen.orientation?.addEventListener(
      "change",
      scheduleOrientation,
    );
    return () => {
      window.removeEventListener("resize", scheduleOrientation);
      window.screen.orientation?.removeEventListener(
        "change",
        scheduleOrientation,
      );
      if (orientationTimerRef.current) {
        clearTimeout(orientationTimerRef.current);
        orientationTimerRef.current = null;
      }
    };
  }, []);

  useEffect(
    () => () => {
      if (runningRef.current) {
        saveVisionSession(
          modeRef.current,
          activeDeviceIdRef.current,
          true,
        );
      }
      stopCamera({ clearSession: false, resetUi: false });
      if (backgroundReleaseTimerRef.current) {
        clearTimeout(backgroundReleaseTimerRef.current);
        backgroundReleaseTimerRef.current = null;
      }
      obstacleWorkerRef.current?.postMessage({ type: "dispose" });
      obstacleWorkerRef.current?.terminate();
      obstacleWorkerRef.current = null;
      obstacleWorkerPromiseRef.current = null;
      obstacleModelReadyRef.current = false;
      void audioContextRef.current?.close();
    },
    [stopCamera],
  );

  const liveStatus =
    cameraState === "paused"
      ? "รอแตะเพื่อเปิดกล้องต่อ"
      : cameraState === "resuming"
      ? "กำลังกู้คืนกล้อง…"
      : modelState === "ready"
      ? `${modelBackend === "webgpu" ? "GPU" : "WASM"} • ${fps ? fps.toFixed(1) : "—"} FPS${runtimeLabel ? ` • ${runtimeLabel}` : ""}`
      : modelState === "loading"
        ? `โหลด AI ${modelProgress}%`
        : modelState === "error"
          ? "AI ไม่พร้อม"
          : "เตรียม AI";
  const gpsLabel =
    gpsState === "ready"
      ? `${speedKmh} กม./ชม.`
      : gpsState === "locating"
        ? "กำลังหา GPS"
        : gpsState === "denied"
          ? "GPS ถูกปิด"
          : gpsState === "unsupported"
            ? "ไม่รองรับ GPS"
            : "สัญญาณ GPS อ่อน";

  return (
    <main
      className={`app-shell ${isRunning ? "is-live" : ""}`}
      data-controls={controlsVisible ? "visible" : "hidden"}
      data-alert={alert?.level ?? "none"}
      onPointerDown={revealControls}
      style={{ "--mode-accent": currentMode.accent } as React.CSSProperties}
    >
      <div className="ambient" aria-hidden="true">
        <div className="ambient-orb ambient-orb-one" />
        <div className="ambient-orb ambient-orb-two" />
        <div className="road-grid" />
      </div>

      <video
        ref={videoRef}
        className="camera-feed"
        autoPlay
        muted
        playsInline
        aria-label="ภาพสดจากกล้องด้านหลัง"
      />
      <canvas
        ref={canvasRef}
        className="vision-layer"
        aria-label="กรอบวัตถุ เลนถนนที่ตรวจพบ ขอบตัวรถ และการประเมินความเสี่ยง"
      />
      <div className="camera-vignette" aria-hidden="true" />

      <header className="top-bar">
        <div className="brand">
          <span className="brand-mark">
            <ShieldCheck size={20} strokeWidth={2.4} />
          </span>
          <span>
            <strong>RoadGuard</strong>
            <small>FAST OBSTACLE AI</small>
          </span>
        </div>

        {isRunning ? (
          <div
            className={`live-status model-${modelState}`}
            role="status"
            aria-live="polite"
          >
            {modelState === "loading" ? (
              <LoaderCircle className="spin" size={13} />
            ) : (
              <span className="status-dot" />
            )}
            {liveStatus}
          </div>
        ) : (
          <div className="privacy-pill">AI • บนเครื่อง</div>
        )}
      </header>

      {!sessionChecked && (
        <section className="welcome-panel boot-panel" role="status">
          <span className="loader-ring">
            <LoaderCircle className="spin" size={25} />
          </span>
          <h1>กำลังคืนสถานะ</h1>
          <p className="lead">ตรวจสอบกล้องและโหมดล่าสุดบนเครื่องนี้</p>
        </section>
      )}

      {sessionChecked && !isRunning && (
        <section className="welcome-panel">
          <div className="eyebrow">
            <span className="eyebrow-dot" />
            AI ROAD AWARENESS
          </div>
          <h1>
            ตาที่สอง
            <br />
            <span>ระหว่างเดินทาง</span>
          </h1>
          <p className="lead">
            ใช้กล้อง iPhone มองวัตถุด้านหน้า
            พร้อมตรวจเลนจริงและการแจ้งเตือนบนจอ
          </p>

          <div className="mode-picker" aria-label="เลือกวิธีเดินทาง">
            {modeEntries.map(([key, item]) => (
              <button
                className={`mode-card ${mode === key ? "is-selected" : ""}`}
                key={key}
                type="button"
                onClick={() => changeMode(key)}
                aria-pressed={mode === key}
              >
                <span className="mode-icon">
                  <ModeIcon mode={key} />
                </span>
                <span>{item.label}</span>
              </button>
            ))}
          </div>

          <p className="mode-copy">{currentMode.copy}</p>

          <button
            className="start-button"
            type="button"
            onClick={() => void startCamera()}
            disabled={cameraState === "requesting"}
          >
            <Camera size={22} />
            {cameraState === "requesting"
              ? "กำลังขอสิทธิ์กล้อง…"
              : cameraState === "error"
                ? "ลองเปิดกล้องอีกครั้ง"
                : "เริ่มกล้อง GPS และ AI"}
          </button>

          <p className="download-note">
            ระบบจะขอสิทธิ์กล้องและตำแหน่ง • ครั้งแรกแนะนำ Wi‑Fi
          </p>

          {errorMessage && (
            <p className="error-message" role="alert">
              {errorMessage}
            </p>
          )}

          <button
            className="safety-link"
            type="button"
            onClick={() => setShowInfo(true)}
          >
            <Info size={16} />
            อ่านก่อนใช้งาน
          </button>
        </section>
      )}

      {(cameraState === "paused" || cameraState === "resuming") && (
        <section className="resume-panel" role="alert">
          <span className="resume-icon">
            {cameraState === "resuming" ? (
              <LoaderCircle className="spin" size={27} />
            ) : (
              <Camera size={27} />
            )}
          </span>
          <div>
            <strong>
              {cameraState === "resuming"
                ? "กำลังขอสิทธิ์กล้อง"
                : "แตะเพื่อเปิดกล้องต่อ"}
            </strong>
            <p>
              {cameraState === "resuming"
                ? "ตอบรับหน้าต่าง Camera ของ Safari หากไม่ขึ้น ระบบจะกลับมาให้ลองใหม่อัตโนมัติ"
                : errorMessage ||
                  "ระบบจำโหมดเดิมไว้แล้ว Safari ต้องได้รับการแตะจากคุณก่อนจึงจะเปิดกล้องได้"}
            </p>
          </div>
          <button
            className="resume-button"
            type="button"
            onClick={() => void resumeCamera()}
            disabled={cameraState === "resuming"}
          >
            {cameraState === "resuming" ? (
              <LoaderCircle className="spin" size={20} />
            ) : (
              <Camera size={20} />
            )}
            {cameraState === "resuming"
              ? "กำลังรอ Safari…"
              : "เปิดกล้องและอนุญาต"}
          </button>
          {cameraState === "paused" && (
            <button
              className="resume-cancel"
              type="button"
              onClick={cancelResume}
            >
              กลับหน้าเริ่มต้น
            </button>
          )}
        </section>
      )}

      {cameraState === "running" && (
        <>
          <div
            className={`mode-badge gps-${gpsState}`}
            role="status"
            aria-label={`${currentMode.label} ${gpsLabel} พบ ${detectedCount} วัตถุ`}
          >
            <ModeIcon mode={mode} size={18} />
            <strong>{currentMode.label}</strong>
            <span className="badge-divider" />
            <Navigation className="gps-icon" size={15} />
            <span className="speed-reading">{gpsLabel}</span>
          </div>

          {modelState === "loading" && (
            <div className="model-loader" role="status" aria-live="polite">
              <span className="loader-ring">
                <LoaderCircle className="spin" size={25} />
              </span>
              <div>
                <strong>กำลังเตรียม Fast Obstacle AI</strong>
                <span>ดาวน์โหลดโมเดล {modelProgress}%</span>
              </div>
              <div className="loader-track" aria-hidden="true">
                <span style={{ width: `${modelProgress}%` }} />
              </div>
            </div>
          )}

          {modelState === "error" && (
            <div className="model-loader model-loader-error" role="alert">
              <span className="loader-ring">
                <AlertTriangle size={23} />
              </span>
              <div>
                <strong>เริ่ม AI ไม่สำเร็จ</strong>
                <span>{modelError || "ตรวจสอบอินเทอร์เน็ตแล้วลองใหม่"}</span>
              </div>
              <button type="button" onClick={retryModel}>
                <RefreshCw size={16} />
                ลองใหม่
              </button>
            </div>
          )}

          {alert && (
            <div
              className={`hazard-alert is-${alert.level}`}
              role="alert"
              aria-live="assertive"
            >
              <span className="hazard-icon">
                <AlertTriangle size={25} strokeWidth={2.4} />
              </span>
              <span className="hazard-copy">
                <strong>{alert.title}</strong>
                <small>{alert.detail}</small>
              </span>
              <span className="hazard-distance">
                {formatDistance(alert.distance)}
              </span>
            </div>
          )}

          <div className="estimate-note">
            GPS ระยะ และเวลาปะทะเป็นค่าประมาณ
          </div>

          <div className="bottom-controls">
            <div className="quick-modes" aria-label="เปลี่ยนโหมดเดินทาง">
              {modeEntries.map(([key, item]) => (
                <button
                  key={key}
                  className={mode === key ? "is-active" : ""}
                  type="button"
                  onClick={() => changeMode(key)}
                  aria-label={`โหมด${item.label}`}
                  aria-pressed={mode === key}
                >
                  <ModeIcon mode={key} size={20} />
                  <span>{item.label}</span>
                </button>
              ))}
            </div>

            <div className="utility-controls">
              <button
                type="button"
                onClick={() => void toggleSound()}
                aria-label={soundEnabled ? "ปิดเสียงเตือน" : "เปิดเสียงเตือน"}
              >
                {soundEnabled ? <Volume2 size={21} /> : <VolumeX size={21} />}
              </button>
              <button
                type="button"
                onClick={() => setShowInfo(true)}
                aria-label="ข้อมูลความปลอดภัย"
              >
                <Info size={21} />
              </button>
              {cameras.length > 1 && (
                <button
                  type="button"
                  onClick={() => void cycleCamera()}
                  aria-label="เปลี่ยนเลนส์กล้อง"
                >
                  <Camera size={21} />
                </button>
              )}
              <button
                type="button"
                onClick={() => void enterFullscreen()}
                aria-label="แสดงเต็มจอ"
              >
                <Maximize2 size={21} />
              </button>
            </div>
          </div>
        </>
      )}

      {showInfo && (
        <div className="sheet-backdrop" role="presentation">
          <section
            className="info-sheet"
            role="dialog"
            aria-modal="true"
            aria-labelledby="safety-title"
          >
            <button
              className="sheet-close"
              type="button"
              onClick={() => setShowInfo(false)}
              aria-label="ปิด"
            >
              <X size={21} />
            </button>
            <span className="sheet-icon">
              <ShieldCheck size={28} />
            </span>
            <h2 id="safety-title">ใช้เป็นผู้ช่วยมองเท่านั้น</h2>
            <p>
              RoadGuard เป็นระบบทดลอง ไม่ใช่ ADAS
              และอาจมองพลาดหรือเตือนช้าได้ ห้ามใช้แทนการมองถนน
              การเว้นระยะ หรือการตัดสินใจของผู้ขับขี่
            </p>
            <ul>
              <li>ยึด iPhone ให้มั่นคงและไม่บังทัศนวิสัย</li>
              <li>อย่าถือหรือแตะหน้าจอขณะขี่หรือขับรถ</li>
              <li>อนุญาตตำแหน่งเพื่อใช้ความเร็ว GPS ปรับระยะเตือน</li>
              <li>GPS และระยะบนจอเป็นค่าประมาณ อาจคลาดเคลื่อนหรือขาดหาย</li>
              <li>ภาพจะแสดงครบทั้งเฟรมที่กล้องส่งมา จึงอาจมีขอบดำเมื่อสัดส่วนหน้าจอไม่ตรงกับกล้อง</li>
              <li>เส้นเลนจะแสดงเฉพาะเมื่อพบขอบเลนซ้ายและขวาด้วยความมั่นใจเพียงพอ</li>
              <li>พื้นที่ใต้ขอบตัวรถที่ตรวจพบจะถูกตัดออก เพื่อลดการตรวจคอนโซลผิดเป็นวัตถุ</li>
              <li>เมื่อ Safari พักหรือโหลดหน้าใหม่ ระบบจะพยายามกลับมาใช้กล้องต่อและลดภาระเป็นโหมดกู้คืนอัตโนมัติ</li>
              <li>บน iPhone ใช้ แชร์ → เพิ่มไปยังหน้าจอโฮม เพื่อเปิดแบบเต็มจอ</li>
            </ul>
            <div className="privacy-box">
              <ShieldCheck size={17} />
              ภาพและข้อมูลตำแหน่งประมวลผลบนอุปกรณ์และไม่ถูกอัปโหลด
            </div>
            <button
              className="sheet-confirm"
              type="button"
              onClick={() => setShowInfo(false)}
            >
              เข้าใจแล้ว
            </button>
          </section>
        </div>
      )}
    </main>
  );
}
