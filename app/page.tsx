"use client";

import type { YOLO as YoloModel } from "@ultralytics/yolo";
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
  analyzeRoadScene,
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

type CameraState = "idle" | "requesting" | "running" | "error";
type ModelState = "idle" | "loading" | "ready" | "error";
type GpsState =
  | "idle"
  | "locating"
  | "ready"
  | "weak"
  | "denied"
  | "unsupported";

const ROAD_CLASS_IDS = [
  0, 1, 2, 3, 5, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
  24, 25, 26, 28, 32, 36,
];

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

function concatenateChunks(chunks: Uint8Array[], totalBytes: number) {
  const output = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

function captureOrientation() {
  return window.innerHeight >= window.innerWidth
    ? ("portrait" as const)
    : ("landscape" as const);
}

function captureShape(
  orientation: "portrait" | "landscape",
): MediaTrackConstraints {
  const portrait = orientation === "portrait";
  return {
    width: {
      ideal: portrait ? 720 : 1280,
      max: portrait ? 720 : 1280,
    },
    height: {
      ideal: portrait ? 1280 : 720,
      max: portrait ? 1280 : 720,
    },
    aspectRatio: { ideal: portrait ? 9 / 16 : 16 / 9 },
    frameRate: { ideal: 24, max: 30 },
  };
}

function cameraConstraints(
  orientation: "portrait" | "landscape",
  deviceId?: string,
): MediaTrackConstraints {
  return {
    ...captureShape(orientation),
    ...(deviceId
      ? { deviceId: { exact: deviceId } }
      : { facingMode: { ideal: "environment" } }),
  };
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inferenceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const roadCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const geolocationWatchRef = useRef<number | null>(null);
  const wakeLockRef = useRef<{ release: () => Promise<void> } | null>(null);
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const modelRef = useRef<YoloModel | null>(null);
  const modelPromiseRef = useRef<Promise<void> | null>(null);
  const runningRef = useRef(false);
  const cameraSessionRef = useRef(0);
  const modeRef = useRef<TravelMode>("drive");
  const soundEnabledRef = useRef(true);
  const inferenceBusyRef = useRef(false);
  const lastInferenceAtRef = useRef(0);
  const lastOverlayAtRef = useRef(0);
  const lastRoadAnalysisAtRef = useRef(0);
  const inferenceIntervalRef = useRef(260);
  const detectionsRef = useRef<AnalyzedDetection[]>([]);
  const roadSceneRef = useRef<RoadScene>(emptyRoadScene());
  const roadSceneTrackerRef = useRef(createRoadSceneTracker());
  const frameSizeRef = useRef({ width: 0, height: 0 });
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

  const [mode, setMode] = useState<TravelMode>("drive");
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [modelState, setModelState] = useState<ModelState>("idle");
  const [modelProgress, setModelProgress] = useState(0);
  const [modelBackend, setModelBackend] = useState("");
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

  const currentMode = MODES[mode];
  const isRunning = cameraState === "running";

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

  const loadYolo = useCallback(async () => {
    if (modelRef.current) return;
    if (modelPromiseRef.current) return modelPromiseRef.current;

    const task = (async () => {
      setModelState("loading");
      setModelProgress(3);
      setModelError("");
      try {
        const response = await fetch("/models/yolo26n.onnx", {
          cache: "force-cache",
        });
        if (!response.ok || !response.body) {
          throw new Error(`ดาวน์โหลดโมเดลไม่สำเร็จ (${response.status})`);
        }

        const total = Number(response.headers.get("content-length") ?? 0);
        const reader = response.body.getReader();
        const chunks: Uint8Array[] = [];
        let received = 0;
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value) continue;
          chunks.push(value);
          received += value.byteLength;
          const progress = total
            ? Math.round((received / total) * 76)
            : Math.min(76, 8 + Math.round(received / 220_000));
          setModelProgress(Math.max(3, Math.min(76, progress)));
        }

        if (!received) throw new Error("ไฟล์โมเดลว่างเปล่า");
        setModelProgress(82);
        const modelBytes = concatenateChunks(chunks, received);
        const { YOLO } = await import("@ultralytics/yolo");
        setModelProgress(91);
        const model = await YOLO.load(modelBytes, { device: "auto" });
        modelRef.current = model;
        setModelBackend(model.device);
        setModelProgress(100);
        setModelState("ready");
      } catch (error) {
        console.error(error);
        const message =
          error instanceof Error
            ? error.message
            : "ไม่สามารถเริ่มโมเดล YOLO ได้";
        setModelError(message);
        setModelState("error");
        modelPromiseRef.current = null;
        throw error;
      }
    })();
    modelPromiseRef.current = task;
    return task;
  }, []);

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

  const stopCamera = useCallback(() => {
    runningRef.current = false;
    cameraSessionRef.current += 1;
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    stopLocationTracking();
    void wakeLockRef.current?.release();
    wakeLockRef.current = null;
    detectionsRef.current = [];
    roadSceneRef.current = emptyRoadScene();
    roadSceneTrackerRef.current = createRoadSceneTracker();
    frameSizeRef.current = { width: 0, height: 0 };
    alertRef.current = null;
    alertHoldRef.current = { alert: null, lastSeenAt: 0 };
    trackStoreRef.current = { nextId: 1, tracks: new Map() };
    setAlert(null);
    setDetectedCount(0);
    setFps(0);
  }, [stopLocationTracking]);

  const analyzeCurrentRoad = useCallback((analyzedAt: number) => {
    const video = videoRef.current;
    if (
      modeRef.current !== "drive" ||
      !video ||
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      !video.videoWidth ||
      !video.videoHeight
    ) {
      roadSceneRef.current = emptyRoadScene(analyzedAt);
      roadSceneTrackerRef.current = createRoadSceneTracker();
      return;
    }

    try {
      const analysisCanvas =
        roadCanvasRef.current ?? document.createElement("canvas");
      roadCanvasRef.current = analysisCanvas;
      const scale = Math.min(
        1,
        320 / Math.max(video.videoWidth, video.videoHeight),
      );
      const width = Math.max(96, Math.round(video.videoWidth * scale));
      const height = Math.max(96, Math.round(video.videoHeight * scale));
      if (
        analysisCanvas.width !== width ||
        analysisCanvas.height !== height
      ) {
        analysisCanvas.width = width;
        analysisCanvas.height = height;
        roadSceneTrackerRef.current = createRoadSceneTracker();
      }
      const context = analysisCanvas.getContext("2d", {
        alpha: false,
        willReadFrequently: true,
      });
      if (!context) return;
      context.drawImage(video, 0, 0, width, height);
      const frame = context.getImageData(0, 0, width, height);
      roadSceneRef.current = analyzeRoadScene(
        frame.data,
        width,
        height,
        roadSceneTrackerRef.current,
        analyzedAt,
      );
    } catch (error) {
      console.warn("Road scene analysis failed", error);
      roadSceneRef.current = emptyRoadScene(analyzedAt);
      roadSceneTrackerRef.current = createRoadSceneTracker();
    }
  }, []);

  const runInference = useCallback(
    async (session: number) => {
      const model = modelRef.current;
      const video = videoRef.current;
      if (
        !model ||
        !video ||
        inferenceBusyRef.current ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        document.hidden
      ) {
        return;
      }

      inferenceBusyRef.current = true;
      const startedAt = performance.now();
      try {
        const inferenceCanvas =
          inferenceCanvasRef.current ?? document.createElement("canvas");
        inferenceCanvasRef.current = inferenceCanvas;
        const sourceScale = Math.min(
          1,
          640 / Math.max(video.videoWidth, video.videoHeight),
        );
        const sourceWidth = Math.max(
          1,
          Math.round(video.videoWidth * sourceScale),
        );
        const sourceHeight = Math.max(
          1,
          Math.round(video.videoHeight * sourceScale),
        );
        if (
          inferenceCanvas.width !== sourceWidth ||
          inferenceCanvas.height !== sourceHeight
        ) {
          inferenceCanvas.width = sourceWidth;
          inferenceCanvas.height = sourceHeight;
        }
        const inferenceContext = inferenceCanvas.getContext("2d", {
          alpha: false,
        });
        if (!inferenceContext) return;
        inferenceContext.drawImage(
          video,
          0,
          0,
          sourceWidth,
          sourceHeight,
        );

        const results = await model.predict(inferenceCanvas, {
          conf: 0.36,
          iou: 0.52,
          classes: ROAD_CLASS_IDS,
        });
        if (
          session !== cameraSessionRef.current ||
          !runningRef.current
        ) {
          return;
        }

        const resultWidth = results.width || sourceWidth;
        const resultHeight = results.height || sourceHeight;
        const scaleX = video.videoWidth / resultWidth;
        const scaleY = video.videoHeight / resultHeight;
        const displayBoxes = results.boxes.map((box) => ({
          ...box,
          x1: box.x1 * scaleX,
          x2: box.x2 * scaleX,
          y1: box.y1 * scaleY,
          y2: box.y2 * scaleY,
        }));
        const motion = freshMotion(motionRef.current);
        if (!motion.reliable && motionRef.current.reliable) {
          motionRef.current = motion;
          setGpsState("weak");
        }
        const now = performance.now();
        const analyzed = analyzeDetections(
          displayBoxes,
          video.videoWidth,
          video.videoHeight,
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

        const elapsed = performance.now() - startedAt;
        const moving = !motion.reliable || motion.speedMps >= 1.2;
        const intervalFloor =
          model.device === "webgpu"
            ? moving
              ? 110
              : 260
            : moving
              ? 240
              : 560;
        inferenceIntervalRef.current = Math.min(
          moving ? 900 : 1200,
          Math.max(
            intervalFloor,
            elapsed * (model.device === "webgpu" ? 0.25 : 0.34),
          ),
        );

        const stats = statsRef.current;
        if (!stats.startedAt) stats.startedAt = performance.now();
        stats.frames += 1;
        const statsElapsed = performance.now() - stats.startedAt;
        if (statsElapsed >= 1600) {
          setFps((stats.frames * 1000) / statsElapsed);
          statsRef.current = { startedAt: performance.now(), frames: 0 };
        }
      } catch (error) {
        console.error("YOLO inference failed", error);
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
          const dimensionsChanged =
            frameSizeRef.current.width !== video.videoWidth ||
            frameSizeRef.current.height !== video.videoHeight;
          if (dimensionsChanged) {
            frameSizeRef.current = {
              width: video.videoWidth,
              height: video.videoHeight,
            };
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            trackStoreRef.current = { nextId: 1, tracks: new Map() };
            roadSceneRef.current = emptyRoadScene(elapsed);
            roadSceneTrackerRef.current = createRoadSceneTracker();
            detectionsRef.current = [];
            alertRef.current = null;
            alertHoldRef.current = { alert: null, lastSeenAt: 0 };
            setDetectedCount(0);
            setAlert(null);
          }

          const roadInterval =
            motionRef.current.reliable &&
            motionRef.current.speedMps < 0.8
              ? 320
              : 190;
          if (
            modeRef.current === "drive" &&
            elapsed - lastRoadAnalysisAtRef.current >= roadInterval
          ) {
            lastRoadAnalysisAtRef.current = elapsed;
            analyzeCurrentRoad(elapsed);
          }

          if (elapsed - lastOverlayAtRef.current >= 33) {
            lastOverlayAtRef.current = elapsed;
            renderVisionOverlay(
              canvas,
              modeRef.current,
              elapsed,
              detectionsRef.current,
              alertRef.current,
              roadSceneRef.current,
            );
          }
          if (
            modelRef.current &&
            elapsed - lastInferenceAtRef.current >=
              inferenceIntervalRef.current &&
            !inferenceBusyRef.current
          ) {
            lastInferenceAtRef.current = elapsed;
            void runInference(session);
          }
        }
        animationRef.current = requestAnimationFrame(frame);
      };
      animationRef.current = requestAnimationFrame(frame);
    },
    [analyzeCurrentRoad, runInference],
  );

  const refreshCameraList = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput",
      );
      const rearCameras = videoDevices.filter((device) =>
        /back|rear|environment|หลัง|ultra|wide|tele/i.test(device.label),
      );
      setCameras(rearCameras.length ? rearCameras : videoDevices);
    } catch {
      setCameras([]);
    }
  }, []);

  const startCamera = useCallback(
    async (deviceId?: string) => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setErrorMessage("Safari รุ่นนี้ไม่รองรับการเปิดกล้องผ่านเว็บ");
        setCameraState("error");
        return;
      }

      await unlockAudio();
      stopCamera();
      const session = ++cameraSessionRef.current;
      lastInferenceAtRef.current = 0;
      lastOverlayAtRef.current = 0;
      lastRoadAnalysisAtRef.current = 0;
      setCameraState("requesting");
      setErrorMessage("");
      void loadYolo().catch(() => undefined);
      try {
        const orientation = captureOrientation();
        captureOrientationRef.current = orientation;
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: cameraConstraints(orientation, deviceId),
        });
        if (session !== cameraSessionRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) throw new Error("camera-view-missing");
        video.srcObject = stream;
        await video.play();
        startLocationTracking();
        await refreshCameraList();
        await requestWakeLock();
        runningRef.current = true;
        setCameraState("running");
        if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
        hideTimerRef.current = setTimeout(
          () => setControlsVisible(false),
          5000,
        );
        beginVisionLoop(session);
      } catch (error) {
        console.error(error);
        stopLocationTracking();
        setErrorMessage(
          "เปิดกล้องไม่ได้ กรุณาอนุญาต Camera ใน Safari แล้วลองอีกครั้ง",
        );
        setCameraState("error");
      }
    },
    [
      beginVisionLoop,
      loadYolo,
      refreshCameraList,
      requestWakeLock,
      startLocationTracking,
      stopCamera,
      stopLocationTracking,
      unlockAudio,
    ],
  );

  const cycleCamera = useCallback(async () => {
    if (cameras.length < 2) return;
    const nextIndex = (cameraIndex + 1) % cameras.length;
    setCameraIndex(nextIndex);
    await startCamera(cameras[nextIndex]?.deviceId);
  }, [cameraIndex, cameras, startCamera]);

  const toggleSound = useCallback(async () => {
    const next = !soundEnabledRef.current;
    if (next) await unlockAudio();
    soundEnabledRef.current = next;
    setSoundEnabled(next);
    if (!next) window.speechSynthesis?.cancel();
  }, [unlockAudio]);

  const retryModel = useCallback(() => {
    modelPromiseRef.current = null;
    void loadYolo().catch(() => undefined);
  }, [loadYolo]);

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

  useEffect(() => {
    const onVisibilityChange = () => {
      if (!document.hidden && runningRef.current) void requestWakeLock();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () =>
      document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [requestWakeLock]);

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

      const track = streamRef.current?.getVideoTracks()[0];
      if (track) {
        void track.applyConstraints(captureShape(orientation)).catch(() => {
          // iOS may rotate the stream itself even when constraints are fixed.
        });
      }
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
      stopCamera();
      modelRef.current?.free();
      modelRef.current = null;
      void audioContextRef.current?.close();
    },
    [stopCamera],
  );

  const liveStatus =
    modelState === "ready"
      ? `${modelBackend === "webgpu" ? "GPU" : "CPU"} • ${fps ? fps.toFixed(1) : "—"} FPS`
      : modelState === "loading"
        ? `โหลด YOLO ${modelProgress}%`
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
            <small>YOLO26 VISION</small>
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
          <div className="privacy-pill">YOLO26 • บนเครื่อง</div>
        )}
      </header>

      {!isRunning && (
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

      {isRunning && (
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
                <strong>กำลังเตรียม YOLO26</strong>
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
              <li>เส้นเลนจะแสดงเฉพาะเมื่อพบขอบเลนซ้ายและขวาด้วยความมั่นใจเพียงพอ</li>
              <li>พื้นที่ใต้ขอบตัวรถที่ตรวจพบจะถูกตัดออก เพื่อลดการตรวจคอนโซลผิดเป็นวัตถุ</li>
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
