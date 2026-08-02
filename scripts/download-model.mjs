import { createHash } from "node:crypto";
import {
  copyFile,
  mkdir,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";

const MODEL_URL =
  "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.onnx";
const EXPECTED_BYTES = 9_941_957;
const EXPECTED_SHA256 =
  "2e947b787d9e787b93a16772a5f55b1d4d8c4d86f53146149c5d6a642442d6f7";
const modelDirectory = path.join(process.cwd(), "public", "models");
const modelPath = path.join(modelDirectory, "yolo26n.onnx");
const temporaryPath = `${modelPath}.download`;
const ortDirectory = path.join(process.cwd(), "public", "ort");
const ortSourceDirectory = path.join(
  process.cwd(),
  "node_modules",
  "onnxruntime-web",
  "dist",
);

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function isValidModel(filePath) {
  try {
    const file = await stat(filePath);
    if (file.size !== EXPECTED_BYTES) return false;
    const bytes = await readFile(filePath);
    return digest(bytes) === EXPECTED_SHA256;
  } catch {
    return false;
  }
}

async function copyOrtRuntime() {
  await mkdir(ortDirectory, { recursive: true });
  await copyFile(
    path.join(ortSourceDirectory, "ort-wasm-simd-threaded.wasm"),
    path.join(ortDirectory, "ort-wasm-simd-threaded.wasm"),
  );
}

await copyOrtRuntime();

if (await isValidModel(modelPath)) {
  console.log("YOLO26 model is ready.");
  process.exit(0);
}

await mkdir(modelDirectory, { recursive: true });
await rm(temporaryPath, { force: true });

const response = await fetch(MODEL_URL, {
  redirect: "follow",
  headers: {
    Accept: "application/octet-stream",
    "User-Agent": "RoadGuard-AI-Build/1.0",
  },
});

if (!response.ok) {
  throw new Error(`Unable to download YOLO26 model (${response.status}).`);
}

const bytes = Buffer.from(await response.arrayBuffer());
if (bytes.byteLength !== EXPECTED_BYTES || digest(bytes) !== EXPECTED_SHA256) {
  throw new Error("YOLO26 model integrity check failed.");
}

await writeFile(temporaryPath, bytes);
await rename(temporaryPath, modelPath);
console.log(`Downloaded YOLO26 model (${bytes.byteLength} bytes).`);
