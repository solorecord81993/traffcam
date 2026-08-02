import * as ort from "onnxruntime-web/webgpu";

const MODEL_INPUT_SIZE = 640;
const SCORE_THRESHOLD = 0.3;
const MAX_RESULTS = 24;
const PAD_VALUE = 114 / 255;

type InitMessage = {
  type: "init";
  modelUrl: string;
};

type InferMessage = {
  type: "infer";
  requestId: number;
  width: number;
  height: number;
  pixels: ArrayBuffer;
};

type DisposeMessage = { type: "dispose" };
type WorkerMessage = InitMessage | InferMessage | DisposeMessage;

type ObstacleBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
  cls: number;
  name: "obstacle";
};

type WorkerScope = {
  onmessage: ((event: MessageEvent<WorkerMessage>) => void) | null;
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
};

const workerScope = self as unknown as WorkerScope;
let session: ort.InferenceSession | null = null;
let backend = "wasm";
let loading: Promise<void> | null = null;

function post(message: unknown, transfer: Transferable[] = []) {
  workerScope.postMessage(message, transfer);
}

function canTryWebGpu() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

async function createSession(modelUrl: string) {
  // Keep the Emscripten JS and WASM files on the same-origin public path.
  // ORT's WebGPU bundle otherwise tries to dynamically import its asyncify
  // runtime next to the hashed worker chunk, which is not a public asset on
  // Vercel and leaves the WASM fallback with no available backend.
  ort.env.wasm.wasmPaths = {
    mjs: "/ort/ort-wasm-simd-threaded.mjs",
    wasm: "/ort/ort-wasm-simd-threaded.wasm",
  };
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads =
    typeof crossOriginIsolated !== "undefined" && crossOriginIsolated
      ? Math.min(4, Math.max(1, Math.ceil((navigator.hardwareConcurrency || 2) / 2)))
      : 1;

  if (canTryWebGpu()) {
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
      backend = "webgpu";
      return;
    } catch (error) {
      console.warn("WebGPU unavailable; falling back to WASM", error);
    }
  }

  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  backend = "wasm";
}

function preprocess(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
) {
  const scale = Math.min(
    MODEL_INPUT_SIZE / Math.max(1, width),
    MODEL_INPUT_SIZE / Math.max(1, height),
  );
  const resizedWidth = Math.max(1, Math.round(width * scale));
  const resizedHeight = Math.max(1, Math.round(height * scale));
  const padX = (MODEL_INPUT_SIZE - resizedWidth) / 2;
  const padY = (MODEL_INPUT_SIZE - resizedHeight) / 2;
  const planeSize = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
  const input = new Float32Array(planeSize * 3);
  input.fill(PAD_VALUE);
  input.fill(PAD_VALUE, planeSize, planeSize * 2);
  input.fill(PAD_VALUE, planeSize * 2);

  for (let y = 0; y < resizedHeight; y += 1) {
    const sourceY = Math.min(
      height - 1,
      Math.max(0, Math.round((y + 0.5) / scale - 0.5)),
    );
    for (let x = 0; x < resizedWidth; x += 1) {
      const sourceX = Math.min(
        width - 1,
        Math.max(0, Math.round((x + 0.5) / scale - 0.5)),
      );
      const sourceOffset = (sourceY * width + sourceX) * 4;
      const targetOffset =
        (Math.floor(padY) + y) * MODEL_INPUT_SIZE + Math.floor(padX) + x;
      input[targetOffset] = pixels[sourceOffset] / 255;
      input[planeSize + targetOffset] = pixels[sourceOffset + 1] / 255;
      input[planeSize * 2 + targetOffset] = pixels[sourceOffset + 2] / 255;
    }
  }

  return {
    tensor: new ort.Tensor("float32", input, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    scale,
    padX,
    padY,
  };
}

function decodeOutput(
  output: ort.Tensor,
  width: number,
  height: number,
  scale: number,
  padX: number,
  padY: number,
) {
  const values = output.data as Float32Array;
  const boxes: ObstacleBox[] = [];
  for (let index = 0; index + 5 < values.length; index += 6) {
    const confidence = Number(values[index + 4]);
    if (!Number.isFinite(confidence) || confidence < SCORE_THRESHOLD) continue;
    const classId = Math.max(0, Math.round(Number(values[index + 5])));
    const x1 = (Number(values[index]) - padX) / scale;
    const y1 = (Number(values[index + 1]) - padY) / scale;
    const x2 = (Number(values[index + 2]) - padX) / scale;
    const y2 = (Number(values[index + 3]) - padY) / scale;
    const left = Math.max(0, Math.min(width, Math.min(x1, x2)));
    const top = Math.max(0, Math.min(height, Math.min(y1, y2)));
    const right = Math.max(0, Math.min(width, Math.max(x1, x2)));
    const bottom = Math.max(0, Math.min(height, Math.max(y1, y2)));
    if (right - left < 2 || bottom - top < 2) continue;
    boxes.push({
      x1: left,
      y1: top,
      x2: right,
      y2: bottom,
      conf: confidence,
      cls: classId,
      name: "obstacle",
    });
  }

  return boxes
    .sort((left, right) => right.conf - left.conf)
    .slice(0, MAX_RESULTS);
}

async function infer(message: InferMessage) {
  if (!session) throw new Error("โมเดลยังไม่พร้อม");
  const pixels = new Uint8ClampedArray(message.pixels);
  const prepared = preprocess(pixels, message.width, message.height);
  const startedAt = performance.now();
  const inputName = session.inputNames[0];
  if (!inputName) throw new Error("โมเดลไม่มี input ที่ใช้งานได้");
  const outputs = await session.run({ [inputName]: prepared.tensor });
  const output = outputs[session.outputNames[0]];
  if (!output) throw new Error("โมเดลไม่ส่งผลลัพธ์");
  const boxes = decodeOutput(
    output,
    message.width,
    message.height,
    prepared.scale,
    prepared.padX,
    prepared.padY,
  );
  post({
    type: "result",
    requestId: message.requestId,
    boxes,
    inferenceMs: performance.now() - startedAt,
  });
}

workerScope.onmessage = (event) => {
  const message = event.data;
  if (message.type === "init") {
    if (!loading) {
      loading = createSession(message.modelUrl)
        .then(() => post({ type: "ready", backend }))
        .catch((error) => {
          loading = null;
          post({
            type: "error",
            stage: "load",
            message: error instanceof Error ? error.message : String(error),
          });
        });
    }
    return;
  }
  if (message.type === "infer") {
    void infer(message).catch((error) => {
      post({
        type: "error",
        stage: "infer",
        requestId: message.requestId,
        message: error instanceof Error ? error.message : String(error),
      });
    });
    return;
  }
  if (message.type === "dispose") {
    void session?.release();
    session = null;
    loading = null;
  }
};
