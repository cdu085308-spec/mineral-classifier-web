(function () {
  "use strict";

  const nativeFetch = window.fetch.bind(window);
  const MODEL_PART_COUNT = window.MINERAL_MODEL_PART_COUNT || 5;
  const MODEL_PART_SIZES = [
    20971520,
    20971520,
    20971520,
    20971520,
    9947815,
  ];
  const MODEL_CACHE_NAME = "mineral-classifier-model-v2";
  const MODEL_CACHE_KEY = "/mineral-classifier-model-v2/model-int8.onnx";

  window.MINERAL_ENGINE = {
    modelLoaded: false,
    modelError: null,
    labels: {},
  };

  if (window.ort) {
    window.ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
    window.ort.env.wasm.numThreads = 1;
    window.ort.env.wasm.proxy = true;
  }

  const dataPromise = nativeFetch("./mineral-data.json").then((response) => {
    if (!response.ok) {
      throw new Error("矿物性质数据库加载失败");
    }
    return response.json();
  });

  async function loadModelBytes() {
    const footer = document.getElementById("footerInfo");
    let modelCache = null;
    if ("caches" in window) {
      modelCache = await window.caches.open(MODEL_CACHE_NAME);
      const cachedModel = await modelCache.match(MODEL_CACHE_KEY);
      if (cachedModel) {
        if (footer) {
          footer.textContent = "正在从浏览器缓存加载识别模型…";
        }
        return new Uint8Array(await cachedModel.arrayBuffer());
      }
    }

    const totalLength = MODEL_PART_SIZES.slice(0, MODEL_PART_COUNT).reduce(
      (total, size) => total + size,
      0,
    );
    const modelBytes = new Uint8Array(totalLength);
    let offset = 0;

    for (let index = 0; index < MODEL_PART_COUNT; index += 1) {
      const expectedSize = MODEL_PART_SIZES[index];
      let partBuffer = null;
      let lastError = null;

      for (let attempt = 1; attempt <= 3; attempt += 1) {
        if (footer) {
          footer.textContent = `正在加载识别模型 ${index + 1}/${MODEL_PART_COUNT}，第 ${attempt} 次尝试…`;
        }
        try {
          const response = await nativeFetch(
            `./models/model-int8.onnx.part${index}?v=2`,
            {
              cache: "force-cache",
              credentials: "omit",
            },
          );
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }
          partBuffer = await response.arrayBuffer();
          if (partBuffer.byteLength !== expectedSize) {
            throw new Error("模型分片大小不完整");
          }
          break;
        } catch (error) {
          lastError = error;
          partBuffer = null;
          await new Promise((resolve) => window.setTimeout(resolve, 1200 * attempt));
        }
      }

      if (!partBuffer) {
        throw new Error(
          `第 ${index + 1} 个模型分片加载失败：${lastError?.message || "网络错误"}`,
        );
      }

      modelBytes.set(new Uint8Array(partBuffer), offset);
      offset += partBuffer.byteLength;
      await new Promise((resolve) => window.setTimeout(resolve, 250));
    }

    if (modelCache) {
      try {
        await modelCache.put(
          MODEL_CACHE_KEY,
          new Response(modelBytes, {
            headers: { "Content-Type": "application/octet-stream" },
          }),
        );
      } catch (error) {
        console.warn("Unable to cache the ONNX model", error);
      }
    }
    return modelBytes;
  }

  const sessionPromise = (async () => {
    if (!window.ort) {
      throw new Error("ONNX Runtime Web 未加载");
    }
    const modelBytes = await loadModelBytes();
    const session = await window.ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    window.MINERAL_ENGINE.modelLoaded = true;
    const footer = document.getElementById("footerInfo");
    if (footer) {
      footer.textContent = "模型已加载 · 支持 282 种矿物识别";
    }
    return session;
  })().catch((error) => {
    window.MINERAL_ENGINE.modelError = error;
    const footer = document.getElementById("footerInfo");
    if (footer) {
      footer.textContent = "模型加载失败，请刷新页面重试";
    }
    throw error;
  });

  const hardnessRanges = {
    "<2.5": [0, 2.5],
    "2.5-3": [2.5, 3],
    "3-5.5": [3, 5.5],
    "5.5-7": [5.5, 7],
    ">7": [7, 10],
  };

  function hardnessMatches(mineralRange, selectedRange) {
    const selected = hardnessRanges[selectedRange];
    if (!selected) return false;
    const overlapLow = Math.max(selected[0], mineralRange[0]);
    const overlapHigh = Math.min(selected[1], mineralRange[1]);
    return overlapHigh - overlapLow >= 0.4;
  }

  function listMatches(values, selectedValue) {
    const selected = new Set(selectedValue.replace(/\s/g, "").split("-"));
    return values.some((value) => {
      const current = new Set(value.replace(/\s/g, "").split("-"));
      return [...selected].some((part) => current.has(part));
    });
  }

  function filterAndScore(data, answers, candidateKeys) {
    let items = Object.entries(data.minerals);
    if (Array.isArray(candidateKeys)) {
      const keys = new Set(
        candidateKeys.map((key) => key.toLowerCase().replace(/\s/g, "-")),
      );
      items = items.filter(([key]) => {
        const normalized = key.toLowerCase().replace(/\s/g, "-");
        return keys.has(normalized) || keys.has(normalized.replace(/-/g, " "));
      });
    }

    const answerEntries = Object.entries(answers);
    const candidates = items.map(([key, properties]) => {
      if (answerEntries.length === 0) {
        return {
          key,
          name_en: key.replace(/-/g, " ").replace(/\b\w/g, (char) => char.toUpperCase()),
          name_cn: properties.name_cn,
          match_score: 100,
          matched: 0,
          total: 0,
        };
      }

      let matched = 0;
      for (const [questionId, selectedValue] of answerEntries) {
        if (
          questionId === "hardness" &&
          hardnessMatches(properties.hardness, selectedValue)
        ) {
          matched += 1;
        } else if (
          ["crystal_system", "luster", "streak", "color", "cleavage", "transparency"].includes(
            questionId,
          ) &&
          listMatches(properties[questionId] || [], selectedValue)
        ) {
          matched += 1;
        } else if (
          questionId === "magnetic" &&
          properties.magnetic === selectedValue
        ) {
          matched += 1;
        }
      }

      return {
        key,
        name_en: key.replace(/-/g, " ").replace(/\b\w/g, (char) => char.toUpperCase()),
        name_cn: properties.name_cn,
        match_score: Math.round((matched / answerEntries.length) * 1000) / 10,
        matched,
        total: answerEntries.length,
      };
    });

    candidates.sort(
      (left, right) =>
        right.match_score - left.match_score || right.matched - left.matched,
    );
    return candidates;
  }

  function propertyWeight(count) {
    return count <= 0 ? 0 : Math.min(0.4 + 0.07 * count, 0.85);
  }

  function computeCombinedScores(candidates, imageScores) {
    if (!imageScores) return candidates;
    const baseWeight = candidates.length ? propertyWeight(candidates[0].total) : 0;
    const scoreMap = new Map(
      Object.entries(imageScores).map(([key, value]) => [
        key.toLowerCase().replace(/\s/g, "-"),
        value,
      ]),
    );

    const output = candidates.map((candidate) => {
      const key = candidate.name_en.toLowerCase().replace(/\s/g, "-");
      const imageConfidence = scoreMap.get(key) || 0;
      let weight = baseWeight;
      let effectiveImage = imageConfidence;

      if (imageConfidence > 40 && candidate.match_score < 80) {
        weight = Math.min(0.9, weight + 0.15);
        const penalty =
          (imageConfidence / 100) * (1 - candidate.match_score / 100);
        effectiveImage = imageConfidence * Math.max(0.05, 1 - penalty * 1.5);
      }
      if (imageConfidence < 10 && candidate.match_score > 85) {
        effectiveImage = 0;
        weight = Math.min(0.9, weight + 0.1);
      }

      return {
        ...candidate,
        image_confidence: Math.round(imageConfidence * 10) / 10,
        combined_score:
          Math.round(
            (weight * candidate.match_score + (1 - weight) * effectiveImage) * 10,
          ) / 10,
      };
    });

    output.sort(
      (left, right) =>
        right.combined_score - left.combined_score ||
        right.match_score - left.match_score,
    );
    return output;
  }

  function nextQuestionId(data, answers) {
    return (
      data.questions.find((question) => !(question.id in answers))?.id || null
    );
  }

  function formatEnglishName(name) {
    return name
      .replace(/-/g, " ")
      .replace(/\b\w/g, (character) => character.toUpperCase());
  }

  async function fileToTensor(file) {
    let imageSource;
    let releaseSource = () => {};
    if (window.createImageBitmap) {
      imageSource = await window.createImageBitmap(file);
      releaseSource = () => imageSource.close?.();
    } else {
      const objectUrl = URL.createObjectURL(file);
      imageSource = await new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("图片读取失败"));
        image.src = objectUrl;
      });
      releaseSource = () => URL.revokeObjectURL(objectUrl);
    }

    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(imageSource, 0, 0, 256, 256);
    releaseSource();

    const pixels = context.getImageData(0, 0, 256, 256).data;
    const tensor = new Float32Array(3 * 256 * 256);
    const means = [0.485, 0.456, 0.406];
    const standardDeviations = [0.229, 0.224, 0.225];
    const planeSize = 256 * 256;

    for (let index = 0; index < planeSize; index += 1) {
      for (let channel = 0; channel < 3; channel += 1) {
        const value = pixels[index * 4 + channel] / 255;
        tensor[channel * planeSize + index] =
          (value - means[channel]) / standardDeviations[channel];
      }
    }

    return new window.ort.Tensor("float32", tensor, [1, 3, 256, 256]);
  }

  async function classifyFile(file, topK = 5) {
    const [data, session] = await Promise.all([dataPromise, sessionPromise]);
    const tensor = await fileToTensor(file);
    const footer = document.getElementById("footerInfo");
    if (footer) {
      footer.textContent = "正在进行本地图像识别，首次通常需要 30-90 秒…";
    }

    let timeoutId;
    const timeoutPromise = new Promise((_, reject) => {
      timeoutId = window.setTimeout(
        () => reject(new Error("识别超时，请重新选择图片或刷新页面")),
        150000,
      );
    });
    let output;
    try {
      output = await Promise.race([
        session.run({ pixel_values: tensor }),
        timeoutPromise,
      ]);
    } finally {
      window.clearTimeout(timeoutId);
    }
    const logits = output.logits.data;

    let maximum = -Infinity;
    for (const value of logits) maximum = Math.max(maximum, value);
    let total = 0;
    const probabilities = new Float32Array(logits.length);
    for (let index = 0; index < logits.length; index += 1) {
      const probability = Math.exp(logits[index] - maximum);
      probabilities[index] = probability;
      total += probability;
    }

    const ranked = Array.from(probabilities.keys())
      .sort((left, right) => probabilities[right] - probabilities[left])
      .slice(0, topK);

    const results = ranked.map((index) => {
      const english = data.labels[String(index)];
      const chinese = data.minerals[english]?.name_cn || english;
      return {
        name_cn: chinese,
        name_en: formatEnglishName(english),
        confidence: Math.round((probabilities[index] / total) * 10000) / 100,
      };
    });
    if (footer) {
      footer.textContent = "识别完成 · 支持 282 种矿物识别";
    }
    return results;
  }

  function jsonResponse(payload, status = 200) {
    return Promise.resolve(
      new Response(JSON.stringify(payload), {
        status,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      }),
    );
  }

  async function handleApi(url, init) {
    try {
      if (url.pathname === "/api/identification/questions") {
        const data = await dataPromise;
        return jsonResponse({
          questions: data.questions,
          total: data.questions.length,
        });
      }

      if (url.pathname === "/api/identification/match") {
        const data = await dataPromise;
        const body = JSON.parse(init?.body || "{}");
        const candidateFilter = body.candidate_filter;
        const imageScores = body.image_scores;
        const answers = {};
        for (const [key, value] of Object.entries(body)) {
          if (key !== "candidate_filter" && key !== "image_scores") {
            answers[key] = value;
          }
        }
        const topK = candidateFilter ? candidateFilter.length * 2 : 15;
        let candidates = filterAndScore(data, answers, candidateFilter).slice(
          0,
          topK,
        );
        if (imageScores) {
          candidates = computeCombinedScores(candidates, imageScores);
        }
        return jsonResponse({
          candidates,
          count: candidates.length,
          next_question_id: nextQuestionId(data, answers),
          filtered: candidateFilter != null,
          has_combined: imageScores != null,
        });
      }

      if (url.pathname === "/api/classes") {
        const data = await dataPromise;
        const classes = Object.values(data.labels).map((english) => ({
          name_cn: data.minerals[english]?.name_cn || english,
          name_en: formatEnglishName(english),
        }));
        return jsonResponse({
          total: classes.length,
          classes,
          loading: false,
        });
      }

      if (url.pathname === "/api/classify") {
        const formData = init?.body;
        const file = formData?.get("file");
        if (!file) {
          return jsonResponse({ detail: "No image file supplied" }, 400);
        }
        const results = await classifyFile(file, 5);
        return jsonResponse({ filename: file.name, results });
      }

      if (url.pathname === "/api/classify-multi") {
        const formData = init?.body;
        const files = formData?.getAll("files") || [];
        if (files.length === 0) {
          return jsonResponse({ detail: "At least one image required" }, 400);
        }
        const scoreBuckets = new Map();
        const filenames = [];
        for (const file of files) {
          filenames.push(file.name);
          const results = await classifyFile(file, 10);
          for (const result of results) {
            if (!scoreBuckets.has(result.name_en)) {
              scoreBuckets.set(result.name_en, []);
            }
            scoreBuckets.get(result.name_en).push(result.confidence);
          }
        }

        const averaged = [...scoreBuckets.entries()]
          .map(([nameEnglish, scores]) => ({
            name_cn:
              dataForNames.minerals?.[
                nameEnglish.toLowerCase().replace(/\s/g, "-")
              ]?.name_cn || "",
            name_en: nameEnglish,
            confidence:
              Math.round(
                (scores.reduce((sum, value) => sum + value, 0) / scores.length) *
                  100,
              ) / 100,
            photos: scores.length,
          }))
          .sort((left, right) => right.confidence - left.confidence);

        return jsonResponse({
          files: filenames,
          results: averaged.slice(0, 10),
          total: filenames.length,
        });
      }

      if (url.pathname === "/api/status") {
        return jsonResponse({
          model_ready: window.MINERAL_ENGINE.modelLoaded,
          classifier_loaded: window.MINERAL_ENGINE.modelLoaded,
          model_error: window.MINERAL_ENGINE.modelError?.message || null,
        });
      }
    } catch (error) {
      return jsonResponse({ detail: error.message || String(error) }, 503);
    }

    return nativeFetch(url, init);
  }

  let dataForNames = {};
  dataPromise.then((data) => {
    dataForNames = data;
    window.MINERAL_ENGINE.labels = data.labels;
  });

  window.fetch = function (input, init) {
    const requestUrl =
      typeof input === "string"
        ? new URL(input, window.location.href)
        : new URL(input.url, window.location.href);
    if (requestUrl.pathname.startsWith("/api/")) {
      return handleApi(requestUrl, init);
    }
    return nativeFetch(input, init);
  };
})();
