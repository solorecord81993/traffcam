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
  RefreshCw,
  ShieldCheck,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
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
  const inferenceIntervalRef = useRef(260);
  const detectionsRef = useRef<AnalyzedDetection[]>([]);
  const alertRef = useRef<VisionAlert | null>(null);
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
    detectionsRef.current = [];
    alertRef.current = null;
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
      const cooldown = nextAlert.level === "danger" ? 2800 : 5200;
      const escalated =
        nextAlert.level === "danger" && previous.level !== "danger";
      if (
        !escalated &&
        now - previous.at < cooldown &&
        previous.key === nextAlert.key
      ) {
        return;
      }
      if (!escalated && now - previous.at < 1900) return;
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

  const stopCamera = useCallback(() => {
    runningRef.current = false;
    cameraSessionRef.current += 1;
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    void wakeLockRef.current?.release();
    wakeLockRef.current = null;
    detectionsRef.current = [];
    alertRef.current = null;
    trackStoreRef.current = { nextId: 1, tracks: new Map() };
    setAlert(null);
    setDetectedCount(0);
    setFps(0);
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
        const results = await model.predict(video, {
          conf: 0.28,
          iou: 0.58,
        });
        if (
          session !== cameraSessionRef.current ||
          !runningRef.current
        ) {
          return;
        }

        const analyzed = analyzeDetections(
          results.boxes,
          results.width || video.videoWidth,
          results.height || video.videoHeight,
          modeRef.current,
          performance.now(),
          trackStoreRef.current,
        );
        const nextAlert = selectAlert(analyzed, modeRef.current);
        detectionsRef.current = analyzed;
        alertRef.current = nextAlert;
        setDetectedCount(analyzed.length);
        setAlert((previous) => {
          if (
            previous?.key === nextAlert?.key &&
            previous?.detail === nextAlert?.detail
          ) {
            return previous;
          }
          return nextAlert;
        });
        if (nextAlert) playWarning(nextAlert);

        const elapsed = performance.now() - startedAt;
        inferenceIntervalRef.current = Math.min(
          950,
          Math.max(model.device === "webgpu" ? 115 : 280, elapsed * 0.72),
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
          if (
            canvas.width !== video.videoWidth ||
            canvas.height !== video.videoHeight
          ) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
          renderVisionOverlay(
            canvas,
            modeRef.current,
            elapsed,
            detectionsRef.current,
            alertRef.current,
          );
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
    [runInference],
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
      setCameraState("requesting");
      setErrorMessage("");
      void loadYolo().catch(() => undefined);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: deviceId
            ? {
                deviceId: { exact: deviceId },
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30, max: 30 },
              }
            : {
                facingMode: { ideal: "environment" },
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30, max: 30 },
              },
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
      stopCamera,
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
        aria-label="กรอบวัตถุ แนวทาง และการประเมินความเสี่ยง"
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
            พร้อมพื้นที่ทางและการแจ้งเตือนบนจอ
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
                : "เริ่มกล้องและ AI"}
          </button>

          <p className="download-note">
            ครั้งแรกจะดาวน์โหลดโมเดล AI และเก็บไว้ในเครื่อง • แนะนำ Wi‑Fi
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
          <div className="mode-badge">
            <ModeIcon mode={mode} size={18} />
            {currentMode.label}
            <span>•</span>
            {modelState === "ready"
              ? `พบ ${detectedCount} วัตถุ`
              : "กำลังเตรียมการตรวจจับ"}
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

          <div className="estimate-note">ระยะและเวลาปะทะเป็นค่าประมาณ</div>

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
              <li>ระยะบนจอเป็นเพียงค่าประมาณจากกล้องเดียว</li>
              <li>YOLO มองวัตถุ แต่แนวทางสีเขียวเป็นทางคาดการณ์จากจุดกึ่งกลางกล้อง</li>
              <li>บน iPhone ใช้ แชร์ → เพิ่มไปยังหน้าจอโฮม เพื่อเปิดแบบเต็มจอ</li>
            </ul>
            <div className="privacy-box">
              <ShieldCheck size={17} />
              ภาพจากกล้องประมวลผลบนอุปกรณ์และไม่ถูกอัปโหลด
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
